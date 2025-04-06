import os
import random
from collections import deque
from typing import Any

# import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Go.Go_uf import UnionFind, get_bit_indices

from plotter import Plotter  # type: ignore
from Buffer import BufferElement

State = np.ndarray[Any, np.dtype[np.int8]]

NUM_POSSIBLE_SCORES = int(30.5 * 2 + 1)


class GlobalPoolingBias(nn.Module):
    def __init__(self, channels_g: int, channels_x: int):
        super().__init__()  # pyright: ignore
        # Pooling layer for G (mean + max)
        pooled_g_features = 2 * channels_g
        # Linear layer to transform pooled G into channel biases for X
        self.fc_bias = nn.Linear(pooled_g_features, channels_x)
        # Batch norm and ReLU applied *before* pooling G, as per Fig 2
        self.bn_g = nn.BatchNorm2d(channels_g)

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        # x is the tensor to be biased (P in policy head) - shape [B, Cx, H, W]
        # g is the tensor providing global context - shape [B, Cg, H, W]

        # Process G: BN -> ReLU
        g_processed = F.relu(self.bn_g(g))

        # Pool processed G
        g_mean = g_processed.mean(dim=(2, 3))  # [B, Cg]
        g_max = g_processed.amax(dim=(2, 3))  # [B, Cg]
        g_pooled = torch.cat([g_mean, g_max], dim=1)  # [B, 2*Cg]

        # Compute channel biases from pooled G
        channel_biases = self.fc_bias(g_pooled)  # [B, Cx]

        # Add biases channel-wise to X
        # Reshape biases to [B, Cx, 1, 1]
        x_biased = x + channel_biases.unsqueeze(-1).unsqueeze(-1)

        return x_biased, g_pooled


class PolicyHead(nn.Module):
    def __init__(self, num_hidden: int, policy_head_channels: int = 16):
        super().__init__()  # pyright: ignore

        # Parallel 1x1 Convs for P and G
        self.conv_p = nn.Conv2d(num_hidden, policy_head_channels, kernel_size=1)
        self.conv_g = nn.Conv2d(num_hidden, policy_head_channels, kernel_size=1)

        # Global Pooling Bias structure
        self.global_bias = GlobalPoolingBias(channels_g=policy_head_channels, channels_x=policy_head_channels)

        # Layers after bias addition (applied to biased P)
        self.bn_p = nn.BatchNorm2d(policy_head_channels)
        self.final_conv = nn.Conv2d(policy_head_channels, 2, kernel_size=1)

        # Linear layer for pass logits (operates on pooled G)
        pooled_g_features = 2 * policy_head_channels
        self.pass_logit_fc = nn.Linear(pooled_g_features, 2)

    def forward(self, x: torch.Tensor):
        # x shape: [B, num_hidden, H, W]

        # Get P and G features
        p = self.conv_p(x)  # [B, chead, H, W]
        g = self.conv_g(x)  # [B, chead, H, W]

        # Apply Global Pooling Bias
        # This processes G, pools it, computes biases, adds them to P
        p_biased, g_pooled = self.global_bias(p, g)  # p_biased=[B, chead, H, W], g_pooled=[B, 2*chead]

        # Process biased P for spatial logits
        p_processed = self.final_conv(F.relu(self.bn_p(p_biased)))  # [B, 2, H, W]

        # Compute pass logits from pooled G
        pass_logits = self.pass_logit_fc(g_pooled)  # [B, 2]

        # Combine spatial and pass logits
        own_policy_spatial = p_processed[:, 0, :, :].reshape(x.size(0), -1)  # [B, H*W]
        opp_policy_spatial = p_processed[:, 1, :, :].reshape(x.size(0), -1)  # [B, H*W]

        own_policy_logits = torch.cat([own_policy_spatial, pass_logits[:, 0:1]], dim=1)  # [B, H*W+1]
        opp_policy_logits = torch.cat([opp_policy_spatial, pass_logits[:, 1:2]], dim=1)  # [B, H*W+1]

        return own_policy_logits, opp_policy_logits


class FinalScoreDistHead(nn.Module):
    def __init__(self, pooled_features_dim: int, value_head_intermediate_channels: int):
        super().__init__()  # pyright: ignore

        self.min_score = -30.5
        self.max_score = 30.5
        self.num_possible_scores = NUM_POSSIBLE_SCORES

        # Layers processing pooled features + score info (shared weights across scores)
        # Input: pooled_features_dim + 2 (scaled score s, parity(s))
        self.fc1 = nn.Linear(pooled_features_dim + 1, value_head_intermediate_channels)
        self.fc2 = nn.Linear(value_head_intermediate_channels, 1)  # Output 1 logit component per score

        # Scaling component (computes gamma)
        self.scale_fc1 = nn.Linear(pooled_features_dim, value_head_intermediate_channels)
        self.scale_fc2 = nn.Linear(value_head_intermediate_channels, 1)  # Output gamma

        self.possible_scores = torch.linspace(
            self.min_score,
            self.max_score,
            self.num_possible_scores,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def forward(self, v_pooled: torch.Tensor) -> torch.Tensor:
        # v_pooled shape: [batch, pooled_features_dim]

        # Compute scaling factor gamma
        gamma = self.scale_fc2(F.relu(self.scale_fc1(v_pooled)))  # [batch, 1]

        # Prepare score features and process each possible score
        # score_logits_list: list[torch.Tensor] = []

        scaled_scores_feat = (self.possible_scores * 0.05).view(1, -1, 1).expand(v_pooled.size(0), -1, -1)
        v_pooled_expanded = v_pooled.unsqueeze(1).expand(-1, self.num_possible_scores, -1)
        combined = torch.cat([v_pooled_expanded, scaled_scores_feat], dim=2)
        unscaled_logits = self.fc2(F.relu(self.fc1(combined)))
        scaled_logits = (unscaled_logits * F.softplus(gamma.unsqueeze(-1))).squeeze(-1)

        # for s in possible_scores:
        #     # Shape: [batch, 1]
        #     score_feat_s = torch.zeros(v_pooled.size(0), 1, device=v_pooled.device)
        #     score_feat_s[:, 0] = 0.05 * s  # Scaled score

        #     # Concatenate pooled features with score features
        #     # Shape: [batch, pooled_features_dim + 2]
        #     combined_features = torch.cat([v_pooled, score_feat_s], dim=1)

        #     # Shape: [batch, 1]
        #     logit_component = self.fc2(F.relu(self.fc1(combined_features)))
        #     score_logits_list.append(logit_component)

        # # Shape: [batch, num_possible_scores]
        # all_score_logits = torch.cat(score_logits_list, dim=1)

        # # Apply scaling factor gamma
        # scaled_logits = all_score_logits * F.softplus(gamma)

        return scaled_logits


class ResNet(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        num_res_blocks: int = 2,
        num_hidden: int = 32,
        num_past_steps: int = 2,
    ) -> None:
        super().__init__()  # pyright: ignore
        self.initialConvBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=1  # disabled nodes
                + 2  # current player, opponent
                + 3  # liberties
                + num_past_steps * 2,  # past moves
                out_channels=num_hidden,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.feature_extractor = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_res_blocks)])

        # --- policy head ---
        self.policy_head = PolicyHead(num_hidden)

        # --- value head base ---
        value_base_channels = 16
        self.vh_init_conv = nn.Conv2d(num_hidden, value_base_channels, kernel_size=1)
        self.vh_global_pooling = GlobalPoolingBias(16, 16)

        pooled_features_dim = 2 * value_base_channels  # mean + max

        # --- value sub heads ---
        value_head_intermediate_channels = 8

        self.game_outcome_head = nn.Sequential(
            nn.Linear(pooled_features_dim, value_head_intermediate_channels),
            nn.ReLU(),
            nn.Linear(value_head_intermediate_channels, 4),
        )

        # output shape [batch, 1, board_width, board_height]
        # +1: current ownership, -1: opponent ownership
        self.ownership_head = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1),
            nn.Tanh(),
        )

        self.score_head = FinalScoreDistHead(pooled_features_dim, value_head_intermediate_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.initialConvBlock(x)
        for block in self.feature_extractor:
            x = block(x)

        # policy head
        own_policy_logits, opp_policy_logits = self.policy_head(x)  # shape [batch, 2, w, h]

        # value head
        v_base = self.vh_init_conv(x)  # shape [batch, 16, w, h]
        v_mean = v_base.mean(dim=(2, 3))  # shape [batch, 16]
        v_max = v_base.amax(dim=(2, 3))  # shape [batch, 16]
        v_pooled = torch.cat([v_mean, v_max], dim=1)  # shape [batch, 32]

        # --- game outcome head ---
        game_outcome_logits = self.game_outcome_head(v_pooled)  # shape [batch, 4]
        outcome_distribution = torch.softmax(game_outcome_logits[:, 0:2], dim=1)  # shape [batch, 2]
        score_difference = game_outcome_logits[:, 2:3] * 20  # shape [batch, 1]
        score_std = F.softplus(game_outcome_logits[:, 3:4]) * 20  # shape [batch, 1]

        game_outcome = torch.cat([outcome_distribution, score_difference, score_std], dim=1)  # shape [batch, 4]

        # --- ownership head ---
        ownership_logits = self.ownership_head(x)  # shape [batch, 1, board_width, board_height]

        # --- score head ---
        score_logits = self.score_head(v_pooled)  # shape [batch, num_possible_scores (-27.5, 27.5)]

        return (own_policy_logits, opp_policy_logits, game_outcome, ownership_logits, score_logits)

    def forward_mcts_eval(self, x: torch.Tensor):
        x = self.initialConvBlock(x)
        for block in self.feature_extractor:
            x = block(x)

        own_policy_logits, _ = self.policy_head(x)

        # Compute Value Head Base and Game Outcome Head
        v_base = self.vh_init_conv(x)
        v_mean = v_base.mean(dim=(2, 3))
        v_max = v_base.amax(dim=(2, 3))
        v_pooled = torch.cat([v_mean, v_max], dim=1)
        game_outcome_head_output = self.game_outcome_head(v_pooled)

        outcome_distribution = torch.softmax(game_outcome_head_output[:, 0:2], dim=1)  # shape [batch, 2]

        return own_policy_logits, outcome_distribution


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()  # pyright: ignore
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class TrainingBuffer:
    def __init__(self, capacity: int = 75000):
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor, int, bool, torch.Tensor, float]] = deque(maxlen=capacity)

    def push(
        self,
        pi_mcts: torch.Tensor,
        pi_opp: torch.Tensor,
        outcome: int,
        was_white: bool,
        group: torch.Tensor,
        score: float,
    ):
        self.buffer.append((pi_mcts, pi_opp, outcome, was_white, group, score))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class AlphaZeroAgent:
    def __init__(
        self,
        board_width: int,
        plotter: Plotter,
        lr: float = 3e-4,
        batch_size: int = 128,
        num_past_steps: int = 2,
        wheight_decay: float = 2e-4,
        checkpoint_dir: str = "models/checkpoints",
    ):
        self.board_width = board_width
        self.board_height = board_width
        self.plotter = plotter
        self.batch_size = batch_size
        self.num_past_steps = num_past_steps
        self.checkpoint_dir = checkpoint_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ResNet(
            board_width, board_width, num_res_blocks=6, num_hidden=128, num_past_steps=num_past_steps
        ).to(self.device)
        self.policy_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=wheight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=4000, eta_min=1e-5)

        self.train_buffer = TrainingBuffer()

        self.possible_scores = torch.linspace(  # shape [1, num_possible_scores]
            self.policy_net.score_head.min_score,
            self.policy_net.score_head.max_score,
            self.policy_net.score_head.num_possible_scores,
            device=self.device,
        ).unsqueeze(0)

    def save_checkpoint(self, filename: str):
        """Saves the model and optimizer state to a file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(  # pyright: ignore
            {
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                # "train_buffer": pickle.dumps(self.train_buffer),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")
        self._manage_checkpoints()

    def load_checkpoint(self, filename: str):
        """Loads the model and optimizer state from a file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path)  # pyright: ignore
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # self.train_buffer = pickle.loads(checkpoint["train_buffer"])
        self.policy_net.eval()  # Set to eval mode after loading
        print(f"Checkpoint loaded from {checkpoint_path}")

    def _manage_checkpoints(self):
        """Keeps only the last 10 checkpoints, deleting older ones."""
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint")],
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
            reverse=True,
        )  # Sort by modification time, newest first

        # Delete checkpoints beyond the last 10
        for checkpoint in checkpoints[10:]:
            filepath = os.path.join(self.checkpoint_dir, checkpoint)
            os.remove(filepath)
            print(f"Deleted old checkpoint: {checkpoint}")

    def augment_state(
        self,
        be: BufferElement,
        outcome: int,
        ownership: np.ndarray[Any, np.dtype[np.int8]],
        score: float,
    ):
        """
        creates 7 augmented versions of the provided state, and pushes all 8 versions into the training buffer.
        """
        uf = be.uf
        pi_mcts = be.pi_mcts
        history = be.history
        was_white = be.is_white
        pi_mcts_opp = be.pi_mcts_response
        assert pi_mcts_opp is not None, "pi_mcts_response must be provided for augmentation."

        pi_board = pi_mcts[: self.board_width * self.board_height].reshape(self.board_width, self.board_height)
        pi_pass = pi_mcts[-1]

        pi_opp = pi_mcts_opp[: self.board_width * self.board_height].reshape(self.board_width, self.board_height)
        pi_res_pass = pi_mcts_opp[-1]

        num_non_history_channels = 4  # state, ownership, pi_board, pi_opp
        grouped_tensor = torch.zeros(
            (self.board_width, self.board_height, num_non_history_channels + self.num_past_steps), device=self.device
        )
        grouped_tensor[:, :, 0] = torch.as_tensor(uf.state, device=self.device)
        grouped_tensor[:, :, 1] = torch.as_tensor(ownership, device=self.device)
        grouped_tensor[:, :, 2] = torch.as_tensor(pi_board)
        grouped_tensor[:, :, 3] = torch.as_tensor(pi_opp)

        padded_history = list(history)
        num_mising = self.num_past_steps - len(padded_history)
        if num_mising > 0:
            padding = [np.zeros((self.board_width, self.board_width), dtype=np.int8)] * num_mising
            padded_history.extend(padding)

        final_history_list = padded_history[: self.num_past_steps]
        for i in range(self.num_past_steps):
            grouped_tensor[:, :, num_non_history_channels + i] = torch.as_tensor(
                final_history_list[i], device=self.device
            )

        self.train_buffer.push(pi_mcts, pi_mcts_opp, outcome, was_white, grouped_tensor, score)

        # rotate state
        for rot in range(1, 4):
            rotated_grouped_tensor = torch.rot90(grouped_tensor, rot)

            rotated_pi_board = rotated_grouped_tensor[:, :, 2]
            rotated_opp_board = rotated_grouped_tensor[:, :, 3]

            rotated_pi = torch.cat([rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])
            rotated_opp = torch.cat([rotated_opp_board.flatten(), pi_res_pass.unsqueeze(0)])

            self.train_buffer.push(rotated_pi, rotated_opp, outcome, was_white, rotated_grouped_tensor, score)

        # mirror state
        mirrored_grouped_tensor = torch.fliplr(grouped_tensor)

        mirrored_pi_board = torch.fliplr(pi_board)
        mirrored_pi = torch.cat([mirrored_pi_board.flatten(), pi_pass.unsqueeze(0)])
        mirrored_pi_response_board = torch.fliplr(pi_opp)
        mirrored_pi_response = torch.cat([mirrored_pi_response_board.flatten(), pi_res_pass.unsqueeze(0)])

        self.train_buffer.push(mirrored_pi, mirrored_pi_response, outcome, was_white, mirrored_grouped_tensor, score)

        for rot in range(1, 4):
            mirrored_rotated_group = torch.rot90(mirrored_grouped_tensor, rot)

            mirrored_rotated_pi_board = torch.rot90(mirrored_pi_board, rot)
            mirrored_rotated_pi = torch.cat([mirrored_rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])

            mirrored_rotated_res_board = torch.rot90(mirrored_pi_response_board, rot)
            mirrored_rotated_response = torch.cat([mirrored_rotated_res_board.flatten(), pi_res_pass.unsqueeze(0)])

            self.train_buffer.push(
                mirrored_rotated_pi, mirrored_rotated_response, outcome, was_white, mirrored_rotated_group, score
            )

    def preprocess_state(self, uf: UnionFind, history: list[State], is_white: bool):
        """
        Convert the board (numpy array) into a float tensor of shape [1,8,w,h].
        1 -> 1.0  Black
        2 -> 1.0  White
        3 -> 1.0  Channel 0
        0 -> 0.0  (Empty)

        Channels:
        0: disabled
        1: current players stones
        2: opponent stones
        3: has one liberty
        4: has two liberties
        5: has three liberties

        History:
        4: current past moves
        5: opponent past moves
        6: current past moves
        7: opponent past moves
        """

        liberty_counts = torch.zeros((self.board_width, self.board_height), device=self.device)
        groups = uf.stones[uf.stones != 0]
        for group in groups:
            stones = get_bit_indices(group)
            for stone in stones:
                x, y = (stone // self.board_height, stone % self.board_height)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                        if uf.state[nx, ny] == 0:
                            liberty_counts[x, y] += 1

        num_channels = 6 + 2 * self.num_past_steps
        result = torch.zeros((1, num_channels, self.board_width, self.board_width), device=self.device)
        board_tensor = torch.as_tensor(uf.state, device=self.device)
        lib_count_tensor = torch.as_tensor(liberty_counts, device=self.device)

        result[0, 0] = (board_tensor == 3).float()  # disabled
        if is_white:
            result[0, 1] = (board_tensor == 2).float()  # white
            result[0, 2] = (board_tensor == 1).float()  # black
        else:
            result[0, 2] = (board_tensor == 1).float()  # black
            result[0, 1] = (board_tensor == 2).float()  # white

        result[0, 3] = (lib_count_tensor == 1).float()  # has one liberty
        result[0, 4] = (lib_count_tensor == 2).float()  # has two liberties
        result[0, 5] = (lib_count_tensor == 3).float()  # has three liberties

        # process history
        for past_idx in range(min(self.num_past_steps, len(history))):
            past_step = torch.as_tensor(history[past_idx], device=self.device)
            if is_white:
                result[0, 6 + past_idx * 2] = (past_step == 2).float()  # white
                result[0, 7 + past_idx * 2] = (past_step == 1).float()  # black
            else:
                result[0, 6 + past_idx * 2] = (past_step == 1).float()  # black
                result[0, 7 + past_idx * 2] = (past_step == 2).float()  # black

        return result

    @torch.no_grad()  # pyright: ignore
    def get_actions_eval(
        self,
        uf: UnionFind,
        game_history: list[State],
        color_is_white: bool,
    ) -> tuple[torch.Tensor, float]:
        """
        Select action deterministically for evaluation (no exploration).
        """
        state_tensor = self.preprocess_state(uf, game_history, color_is_white)

        # Get logits
        policy, outcome_logits = self.policy_net.forward_mcts_eval(state_tensor)
        outcome = outcome_logits.squeeze(0)  # shape [2]
        return policy.squeeze(0), outcome[0].item()

    def decode_action(self, action_idx: int) -> tuple[int, int]:
        """
        Convert the action index back to (x, y) or 'pass'.
        If action_idx == board_size, then 'pass'.
        """
        board_size = self.board_width * self.board_height
        if action_idx == board_size:
            return (-1, -1)
        else:
            x = action_idx // self.board_height
            y = action_idx % self.board_height
            return (x, y)

    def train_step(self) -> None:
        if len(self.train_buffer) < self.batch_size:
            return

        print("=======================================================================")
        # 1. sample batch from train buffer
        batch = self.train_buffer.sample(self.batch_size)

        # tuple[torch.Tensor, torch.Tensor, int, bool, torch.Tensor]
        (pi_list, pi_opp_list, z_list, is_white_list, group_list, score_list) = zip(*batch)

        # 2. Convert data into tensors
        state_tensor_list: list[torch.Tensor] = []
        ownership_list: list[torch.Tensor] = []
        for i in range(len(batch)):
            current_group = group_list[i]
            uf = UnionFind.get_uf_from_state(current_group[:, :, 0].cpu().numpy(), None)
            history = group_list[i][:, :, 4:]
            history_list = [history[:, :, j] for j in range(self.num_past_steps)]

            t = self.preprocess_state(uf, history_list, is_white_list[i])
            state_tensor_list.append(t)
            ownership_list.append(current_group[:, :, 1])

        state_batch = torch.cat(state_tensor_list)  # shape [B, channels, W, H]
        pi_batch = torch.stack(pi_list, dim=0).to(device=self.device)  # shape [B, 26]
        pi_opp_batch = torch.stack(pi_opp_list, dim=0).to(device=self.device)  # shape [B, 26]
        z_batch = torch.tensor(z_list, dtype=torch.float, device=self.device)  # [B]
        # z_batch_transformed = torch.stack([(z_batch + 1) / 2, (1 - z_batch) / 2], dim=1)  # [B, 2]
        z_class_indices = ((1 - z_batch) / 2).long()

        ownership_batch = torch.stack(ownership_list, dim=0).to(device=self.device)  # shape [B, 1, W, H]
        o_target_player = (ownership_batch == 1).float()
        o_target_opponent = (ownership_batch == -1).float()
        o_target_shared = (ownership_batch == 0).float()
        o_target_prob = torch.stack(
            [
                o_target_player + 0.5 * o_target_shared,  # Prob current player owns
                o_target_opponent + 0.5 * o_target_shared,  # Prob opponent owns
            ],
            dim=1,
        ).to(device=self.device)

        score_batch = torch.tensor(score_list, device=self.device)  # shape [B]

        # 3. feed states trough NN
        # logits shape [B, 26], values shape [B, 1] (assuming your net does that)
        logits_own, logits_opp, outcome_logits, ownership_logits, score_logits = self.policy_net(state_batch)

        # 4. Calculate losses
        policy_log_probs_own = F.log_softmax(logits_own, dim=1)  # shape [B, num_actions]
        policy_log_probs_opp = F.log_softmax(logits_opp, dim=1)  # shape [B, num_actions]
        # z_hat = F.log_softmax(outcome_logits[:, :2], dim=1)  # shape [B, 2]
        score_log_probs = F.log_softmax(score_logits, dim=1)  # shape [B, num_possible_scores]
        score_probs = F.softmax(score_logits, dim=1)  # shape [B, num_possible_scores]
        mu_hat = outcome_logits[:, 2]
        sigma_hat = outcome_logits[:, 3]

        ## ownership loss preparation
        ownership_logits_player = (ownership_logits + 1) / 2  # type: ignore
        ownership_logits_opp = 1 - ownership_logits_player  # type: ignore
        ownership_hat = torch.cat([ownership_logits_player, ownership_logits_opp], dim=1)  # shape [B, 2]
        epsilon = 1e-9
        ownership_hat_prob = torch.clamp(
            ownership_hat, epsilon, 1.0 - epsilon
        )  # move predictions close to 0 and 1 away from it

        B, C, _, _ = o_target_prob.shape
        o_target_flat = o_target_prob.view(B, C, -1)  # shape [B, 2, W*H]
        ownership_hat_flat = ownership_hat_prob.view(B, C, -1)  # shape [B, 2, W*H]

        ## score loss preparation
        score_onehot = torch.zeros((self.batch_size, NUM_POSSIBLE_SCORES), device=self.device)
        for i in range(self.batch_size):
            score_idx = int(NUM_POSSIBLE_SCORES / 2) + torch.floor(score_batch[i])
            score_onehot[i, int(score_idx)] = 1.0

        # calculate losses
        # pi_batch is shape [B, num_actions]
        policy_loss_own = -(pi_batch * policy_log_probs_own).sum(dim=1).mean()  # cross entropy loss
        policy_loss_opp = -(pi_opp_batch * policy_log_probs_opp).sum(dim=1).mean() * 0.15  # cross entropy loss

        # game_outcome_value_loss = -(z_batch_transformed * z_hat).sum(dim=1).mean() * 1.5
        game_outcome_value_loss = F.cross_entropy(outcome_logits[:, :2], z_class_indices) * 1.5

        ownership_loss = -(o_target_flat * torch.log(ownership_hat_flat)).sum(dim=1).mean() * 0.06

        score_pdf_loss = -(score_onehot * score_log_probs).sum(dim=1).mean() * 0.02

        target_cdf = torch.cumsum(score_onehot, dim=1)  # shape [B, num_possible_scores]
        predicted_cdf = torch.cumsum(score_probs, dim=1)  # shape [B, num_possible_scores]
        score_cdf_loss = torch.mean(torch.sum((target_cdf - predicted_cdf) ** 2, dim=1)) * 0.02

        mu_s = torch.sum(self.possible_scores * score_probs, dim=1, keepdim=True)  # shape [B, 1]
        variance_s = torch.sum(((self.possible_scores - mu_s) ** 2) * score_probs, dim=1, keepdim=True)  # shape [B, 1]
        sigma_s = torch.sqrt(variance_s + epsilon)  # epsilon in case variance_s is 0

        score_mean_loss = F.huber_loss(mu_hat, score_batch, delta=10.0) * 0.004
        score_std_loss = F.huber_loss(sigma_hat, sigma_s.detach(), delta=10.0) * 0.004

        loss = (
            policy_loss_own
            + policy_loss_opp
            + game_outcome_value_loss
            + ownership_loss
            + score_pdf_loss
            + score_cdf_loss
            + score_mean_loss
            + score_std_loss
        )

        self.plotter.update_loss(loss.item())
        self.plotter.update_policy_loss(policy_loss_own.item())
        self.plotter.update_value_loss(game_outcome_value_loss.item())
        self.plotter.update_stat("policy_loss_opp", policy_loss_opp.item())  # type: ignore
        self.plotter.update_stat("ownership_loss", ownership_loss.item())  # type: ignore
        self.plotter.update_stat("score_pdf_loss", score_pdf_loss.item())  # type: ignore
        self.plotter.update_stat("score_cdf_loss", score_cdf_loss.item())  # type: ignore
        self.plotter.update_stat("score_mean_loss", score_mean_loss.item())  # type: ignore
        self.plotter.update_stat("score_std_loss", score_std_loss.item())  # type: ignore

        # 5. Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore
        self.scheduler.step()

        current_lr = self.scheduler.get_last_lr()[0]
        print(
            f"Training step: loss={loss}, policy_loss={policy_loss_own}, value_loss={game_outcome_value_loss}, lr={current_lr}"
        )

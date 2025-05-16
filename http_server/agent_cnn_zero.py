import os
from typing import Any
from enum import IntEnum


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Go.Go_uf import UnionFind

from plotter import TensorBoardPlotter  # type: ignore
from Buffer import BufferElement, TrainingBuffer
from go_types import State, StateBatch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MCTS_zero_uf import Node


class GlobalPoolingBias(nn.Module):
    def __init__(self, channels_g: int, channels_x: int):
        super().__init__()  # pyright: ignore
        # Pooling layer for G (mean / linear / max)
        pooled_g_features = 3 * channels_g  # TODO: make variable
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
        g_std = g_processed.std(dim=(2, 3))  # [B, Cg]
        g_pooled = torch.cat([g_mean, g_max, g_std], dim=1)  # [B, 3*Cg]

        # Compute channel biases from pooled G
        channel_biases = self.fc_bias(g_pooled)  # [B, Cx]

        # Add biases channel-wise to X
        # Reshape biases to [B, Cx, 1, 1]
        x_biased = x + channel_biases.unsqueeze(-1).unsqueeze(-1)

        return x_biased, g_pooled


class PolicyHead(nn.Module):
    def __init__(self, num_hidden: int, policy_head_channels: int):
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
        pooled_g_features = 3 * policy_head_channels  # TODO: make variable
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
    def __init__(self, pooled_features_dim: int, value_head_intermediate_channels: int, max_score: float):
        super().__init__()  # pyright: ignore

        self.min_score = -max_score
        self.max_score = max_score
        self.num_possible_scores = int(max_score * 2 + 1)

        # Layers processing pooled features + score info (shared weights across scores)
        # Input: pooled_features_dim + 2 (scaled score s, parity(s))
        self.fc1 = nn.Linear(
            pooled_features_dim + 1, value_head_intermediate_channels
        )  # +1 for (0.05 * s) scaling part
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
        # v_pooled shape: [B, pooled_features_dim]

        # Compute scaling factor gamma
        gamma = self.scale_fc2(F.relu(self.scale_fc1(v_pooled)))  # [B, 1]

        # Prepare score features and process each possible score
        scaled_scores_feat = (self.possible_scores * 0.05).view(1, -1, 1).expand(v_pooled.size(0), -1, -1)
        v_pooled_expanded = v_pooled.unsqueeze(1).expand(-1, self.num_possible_scores, -1)
        combined = torch.cat([v_pooled_expanded, scaled_scores_feat], dim=2)
        unscaled_logits = self.fc2(F.relu(self.fc1(combined)))
        scaled_logits = (unscaled_logits * F.softplus(gamma.unsqueeze(-1))).squeeze(-1)

        return scaled_logits


class ResNet(nn.Module):
    def __init__(
        self,
        num_past_steps: int,
        num_res_blocks: int,
        num_pooling_blocks: int,
        num_hidden: int,
        max_score: float,
    ) -> None:
        super().__init__()  # pyright: ignore

        self.num_input_channels = (
            1  # disabled nodes
            + 2  # current player, opponent
            + 3  # liberties
            + 1  # KO moves
            + num_past_steps  # past moves, one-hot encoded
        )

        self.initialConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=self.num_input_channels, out_channels=num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        # game info vector input
        vector_feature_size = num_past_steps + 4  # Where pass + num own + num opp + num empty locations + you started
        self.vector_processor = nn.Linear(vector_feature_size, num_hidden)

        # spatial game data input
        self.feature_extractor = nn.ModuleList()
        indices = np.linspace(1, num_res_blocks - 2, num_pooling_blocks, dtype=int)
        print(f"Using global pooling in ResBlocks at indices: {list(indices)}")

        for i in range(num_res_blocks):
            if i in indices:
                self.feature_extractor.append(ResBlockGlobal(num_hidden, num_pool_channels=32))
            else:
                self.feature_extractor.append(ResBlock(num_hidden))  # Your original ResBlock

        # --- Final Layer before Heads ---
        self.final_bn = nn.BatchNorm2d(num_hidden)

        # --- Policy head ---
        self.policy_head = PolicyHead(num_hidden, policy_head_channels=48)

        # --- Value head base ---
        value_base_channels = 64
        self.vh_init_conv = nn.Conv2d(num_hidden, value_base_channels, kernel_size=1)

        # --- Value sub heads ---
        pooled_features_dim = 3 * value_base_channels  # mean / mean_linear / max # + mean_quadratic
        value_head_intermediate_channels = 48

        self.game_outcome_head = nn.Sequential(
            nn.Linear(pooled_features_dim, value_head_intermediate_channels),
            nn.ReLU(),
            nn.Linear(
                value_head_intermediate_channels, 3
            ),  # who wins, score mean, score std # win / lose / score mean / score std dev
        )

        # output shape [B, 1, board_width, board_height]
        # +1: current ownership, -1: opponent ownership
        self.ownership_head = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1),
            nn.Tanh(),
        )

        self.score_head = FinalScoreDistHead(pooled_features_dim, value_head_intermediate_channels, max_score)

    def forward(
        self, x: torch.Tensor, x_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [B, 1, board_width, board_height]
        # x_vector shape: [B, 5]

        spatial_features = self.initialConvBlock(x)
        vector_features = self.vector_processor(x_vector)  # shape [B, num_hidden]
        vector_bias_reshaped = vector_features.unsqueeze(-1).unsqueeze(-1)  # shape [B, num_hidden, 1, 1]

        x = spatial_features + vector_bias_reshaped

        for block in self.feature_extractor:
            x = block(x)

        x = F.relu(self.final_bn(x))

        # policy head
        own_policy_logits, opp_policy_logits = self.policy_head(x)  # shape [B, 2, w, h]

        # value head
        v_base = self.vh_init_conv(x)  # shape [B, c_val, w, h]
        v_mean = v_base.mean(dim=(2, 3))  # shape [B, c_val]
        v_max = v_base.amax(dim=(2, 3))  # shape [B, c_val]
        v_std = v_base.std(dim=(2, 3))  # shape [B, c_val]

        v_pooled = torch.cat([v_mean, v_max, v_std], dim=1)  # shape [B, c_val]

        # --- game outcome head ---
        game_outcome_logits = self.game_outcome_head(v_pooled)  # shape [B, 4]
        # outcome_distribution = torch.softmax(game_outcome_logits[:, 0:2], dim=1)  # shape [B, 2]
        value_output = torch.tanh(game_outcome_logits[:, 0:1])  # NEW, shape [B, 1]
        score_mean = game_outcome_logits[:, 1:2] * 20  # shape [B, 1]
        score_std = F.softplus(game_outcome_logits[:, 2:3]) * 20  # shape [B, 1]

        game_outcome = torch.cat([value_output, score_mean, score_std], dim=1)  # shape [B, 3]

        # --- ownership head ---
        ownership_logits = self.ownership_head(x)  # shape [B, 1, board_width, board_height]

        # --- score head ---
        score_logits = self.score_head(v_pooled)  # shape [B, num_possible_scores (-27.5, 27.5)]

        return (own_policy_logits, opp_policy_logits, game_outcome, ownership_logits, score_logits, v_pooled)

    def forward_mcts_eval(self, x: torch.Tensor, x_vector: torch.Tensor):
        spatial_features = self.initialConvBlock(x)
        vector_features = self.vector_processor(x_vector)
        vector_bias_reshaped = vector_features.unsqueeze(-1).unsqueeze(-1)

        x = spatial_features + vector_bias_reshaped
        for block in self.feature_extractor:
            x = block(x)

        x = F.relu(self.final_bn(x))

        own_policy_logits, _ = self.policy_head(x)

        # Compute Value Head Base and Game Outcome Head
        v_base = self.vh_init_conv(x)  # shape [B, c_val, w, h]
        v_mean = v_base.mean(dim=(2, 3))  # shape [B, c_val]
        v_max = v_base.amax(dim=(2, 3))  # shape [B, c_val]
        v_std = v_base.std(dim=(2, 3))  # shape [B, c_val]

        v_pooled = torch.cat([v_mean, v_max, v_std], dim=1)  # shape [B, c_val]
        game_outcome_head_output = self.game_outcome_head(v_pooled)

        # outcome_distribution = torch.softmax(game_outcome_head_output[:, 0:2], dim=1)  # shape [B, 2]
        value_output = torch.tanh(game_outcome_head_output[:, 0:1])  # NEW, shape [B, 1]
        mu = game_outcome_head_output[:, 1:2] * 20  # shape [B, 1]
        sigma = F.softplus(game_outcome_head_output[:, 2:3]) * 20

        return own_policy_logits, value_output, mu, sigma


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()  # pyright: ignore
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x += residual
        return x


class ResBlockGlobal(nn.Module):
    def __init__(self, num_hidden: int, num_pool_channels: int):
        super().__init__()  # pyright: ignore
        assert num_pool_channels > 0 and num_pool_channels < num_hidden

        self.num_hidden = num_hidden
        self.num_pool_channels = num_pool_channels
        self.num_bias_channels = num_hidden - num_pool_channels

        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)

        self.global_bias = GlobalPoolingBias(channels_g=self.num_pool_channels, channels_x=self.num_bias_channels)

        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        residual = x

        intermediate = self.conv1(F.relu(self.bn1(x)))

        g_channels = intermediate[:, : self.num_pool_channels, :, :]  # [B, C_pool, H, W]
        x_channels_to_bias = intermediate[:, self.num_pool_channels :, :, :]  # [B, C_bias, H, W]

        x_biased, _ = self.global_bias(x_channels_to_bias, g_channels)  # [B, C_bias, H, W]

        intermediate_biased = torch.cat([g_channels, x_biased], dim=1)  # [B, C_hidden, H, W]

        out = self.conv2(F.relu(self.bn2(intermediate_biased)))  # [B, C_hidden, H, W]
        out += residual

        return out


class GroupIdx(IntEnum):
    STATE = 0
    OWNERSHIP = 1
    PI_OWN = 2
    PI_OPP = 3
    VALID_MOVES = 4
    HISTORY = 5


class AlphaZeroAgent:
    def __init__(
        self,
        board_width: int,
        komi: float,
        plotter: TensorBoardPlotter | None,
        lr: float = 1e-4,
        batch_size: int = 128,
        num_past_steps: int = 5,
        wheight_decay: float = 3e-5,
        checkpoint_dir: str = "models/checkpoints",
    ):
        self.board_width = board_width
        self.board_height = board_width
        self.plotter = plotter
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_score = board_width * board_width + komi
        self.num_possible_scores = int(self.max_score * 2 + 1)

        self.num_past_steps = num_past_steps
        # plus one for pass move evaluation, the difference between the past steps is checked. For n past steps, need to check n+1 boards
        self.history_length = num_past_steps + 1

        self.policy_net = ResNet(
            num_past_steps=num_past_steps,
            num_res_blocks=6,
            num_hidden=96,
            num_pooling_blocks=2,
            max_score=self.max_score,
        ).to(self.device)
        self.policy_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=wheight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-5)

        self.train_buffer = TrainingBuffer()

        self.possible_scores = torch.linspace(  # shape [1, num_possible_scores]
            -self.max_score, self.max_score, self.num_possible_scores, device=self.device
        ).unsqueeze(0)

        self.liberty_kernel_cpu = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device="cpu", dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        self.liberty_kernel_gpu = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def save_checkpoint(self, filename: str):
        """Saves the model and optimizer state to a file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(  # pyright: ignore
            {
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            checkpoint_path,
        )
        buffer_path = os.path.join(self.checkpoint_dir, "buffer.pkl")
        torch.save({"buffer": self.train_buffer}, buffer_path)  # pyright: ignore
        print(f"Checkpoint saved to {checkpoint_path}")
        self._manage_checkpoints()

    def load_checkpoint(self, filename: str, load_buffer: bool, load_optimizer_scheduler: bool = True):
        """Loads the model and optimizer state from a file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path)  # pyright: ignore
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer_scheduler:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if load_buffer:
            buffer_path = os.path.join(self.checkpoint_dir, "buffer.pkl")
            self.train_buffer = torch.load(buffer_path, weights_only=False)["buffer"]  # pyright: ignore
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
        perspective_adj_outcome: int,
        ownership: np.ndarray[Any, np.dtype[np.int8]],
        score: float,
        white_started: bool,
    ):
        """
        creates 7 augmented versions of the provided state, and pushes all 8 versions into the training buffer.
        """
        uf = be.uf
        pi_mcts = be.pi_mcts
        history = be.history
        valid_moves = be.valid_moves
        is_white = be.is_white

        full_search = be.full_search
        pi_mcts_opp = be.pi_mcts_response
        assert pi_mcts_opp is not None, "pi_mcts_response must be provided for augmentation."

        pi_board = pi_mcts[: self.board_width * self.board_height].reshape(self.board_width, self.board_height)
        pi_pass = pi_mcts[-1]

        pi_opp = pi_mcts_opp[: self.board_width * self.board_height].reshape(self.board_width, self.board_height)
        pi_res_pass = pi_mcts_opp[-1]

        num_non_history_channels = GroupIdx.HISTORY  # state, ownership, pi_board, pi_opp, valid_moves
        # group tensor is channel-last, since fliplr and rot90 expect it that way
        grouped_tensor = torch.zeros(
            (self.board_width, self.board_height, num_non_history_channels + self.history_length), device=self.device
        )
        grouped_tensor[:, :, GroupIdx.STATE] = torch.as_tensor(uf.state, device=self.device)
        grouped_tensor[:, :, GroupIdx.OWNERSHIP] = torch.as_tensor(ownership, device=self.device)
        grouped_tensor[:, :, GroupIdx.PI_OWN] = torch.as_tensor(pi_board)
        grouped_tensor[:, :, GroupIdx.PI_OPP] = torch.as_tensor(pi_opp)
        grouped_tensor[:, :, GroupIdx.VALID_MOVES] = torch.as_tensor(valid_moves, device=self.device)

        padded_history = list(history)
        num_mising = self.history_length - len(padded_history)
        if num_mising > 0:
            padding = [np.zeros((self.board_width, self.board_width), dtype=np.int8)] * num_mising
            padded_history.extend(padding)

        final_history_list = padded_history[: self.history_length]
        for i in range(self.history_length):
            grouped_tensor[:, :, num_non_history_channels + i] = torch.as_tensor(
                final_history_list[i], device=self.device
            )

        self.train_buffer.push(
            pi_mcts, pi_mcts_opp, perspective_adj_outcome, is_white, grouped_tensor, score, full_search, white_started
        )

        # rotate state
        for rot in range(1, 4):
            rotated_grouped_tensor = torch.rot90(grouped_tensor, rot)

            rotated_pi_board = rotated_grouped_tensor[:, :, GroupIdx.PI_OWN]
            rotated_opp_board = rotated_grouped_tensor[:, :, GroupIdx.PI_OPP]

            rotated_pi = torch.cat([rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])
            rotated_opp = torch.cat([rotated_opp_board.flatten(), pi_res_pass.unsqueeze(0)])

            self.train_buffer.push(
                rotated_pi,
                rotated_opp,
                perspective_adj_outcome,
                is_white,
                rotated_grouped_tensor,
                score,
                full_search,
                white_started,
            )

        # mirror state
        mirrored_grouped_tensor = torch.fliplr(grouped_tensor)

        mirrored_pi_board = mirrored_grouped_tensor[:, :, GroupIdx.PI_OWN]
        mirrored_pi_response_board = mirrored_grouped_tensor[:, :, GroupIdx.PI_OPP]

        mirrored_pi = torch.cat([mirrored_pi_board.flatten(), pi_pass.unsqueeze(0)])
        mirrored_pi_response = torch.cat([mirrored_pi_response_board.flatten(), pi_res_pass.unsqueeze(0)])

        self.train_buffer.push(
            mirrored_pi,
            mirrored_pi_response,
            perspective_adj_outcome,
            is_white,
            mirrored_grouped_tensor,
            score,
            full_search,
            white_started,
        )

        for rot in range(1, 4):
            mirrored_rotated_group = torch.rot90(mirrored_grouped_tensor, rot)

            mirrored_rotated_pi_board = mirrored_rotated_group[:, :, GroupIdx.PI_OWN]
            mirrored_rotated_res_board = mirrored_rotated_group[:, :, GroupIdx.PI_OPP]

            mirrored_rotated_pi = torch.cat([mirrored_rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])
            mirrored_rotated_response = torch.cat([mirrored_rotated_res_board.flatten(), pi_res_pass.unsqueeze(0)])

            self.train_buffer.push(
                mirrored_rotated_pi,
                mirrored_rotated_response,
                perspective_adj_outcome,
                is_white,
                mirrored_rotated_group,
                score,
                full_search,
                white_started,
            )

    def preprocess_state(
        self,
        state: State,
        history: list[State],
        valid_moves: np.ndarray[Any, np.dtype[np.bool_]],
        is_white: bool,
        white_started: bool,
        device: str,
    ):
        """
        Convert the board (numpy array) into a float tensor of shape [1,8,w,h].

        Channels:
        0: disabled
        1: current players stones
        2: opponent stones
        3: has one liberty
        4: has two liberties
        5: has three liberties
        6: KO illegal moves

        History:
        7: histroy action one-hot encoded at t-1
        8: histroy action one-hot encoded at t-2
        9: ...
        ...
        """

        num_channels = 7 + self.num_past_steps
        spacial_data_tensor = torch.zeros((1, num_channels, self.board_width, self.board_width), device=device)
        board_tensor = torch.as_tensor(state, device=device)

        # liberty count tensor
        empty_mask = (board_tensor == 0).float().unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        stone_mask = (board_tensor == 1) | (board_tensor == 2)

        kernel = self.liberty_kernel_cpu if device == "cpu" else self.liberty_kernel_gpu
        neighbor_liberties = F.conv2d(empty_mask, kernel, padding=1).squeeze(0).squeeze(0)
        liberty_counts = neighbor_liberties * stone_mask

        # ko move tensor
        illegal_moves_tensor = torch.as_tensor(np.invert(valid_moves), device=device)
        # mask black, white and disabled locations to only KO locations are included
        illegal_moves_tensor[board_tensor == 1] = 0
        illegal_moves_tensor[board_tensor == 2] = 0
        illegal_moves_tensor[board_tensor == 3] = 0

        spacial_data_tensor[0, 0] = (board_tensor == 3).float()  # disabled
        if is_white:
            spacial_data_tensor[0, 1] = (board_tensor == 2).float()  # white
            spacial_data_tensor[0, 2] = (board_tensor == 1).float()  # black
        else:
            spacial_data_tensor[0, 1] = (board_tensor == 1).float()  # black
            spacial_data_tensor[0, 2] = (board_tensor == 2).float()  # white

        spacial_data_tensor[0, 3] = (liberty_counts == 1).float()  # has one liberty
        spacial_data_tensor[0, 4] = (liberty_counts == 2).float()  # has two liberties
        spacial_data_tensor[0, 5] = (liberty_counts == 3).float()  # has three liberties

        spacial_data_tensor[0, 6] = (illegal_moves_tensor == 1).float()  # has three liberties

        # process history from most recent to oldest
        passes = torch.zeros(self.num_past_steps, dtype=torch.int8, device=device)
        num_non_history_channels = self.policy_net.num_input_channels - self.num_past_steps

        # process t and t-1
        if len(history) > 0:
            if np.array_equal(state, history[0]):
                passes[0] = 1
            else:
                diff = history[0] - state
                placed_mask = diff < 0
                spacial_data_tensor[0, num_non_history_channels][placed_mask] = 1

            # process history from t-1 to t-k
            for history_idx in range(1, min(self.num_past_steps, len(history))):
                if np.array_equal(history[history_idx - 1], history[history_idx]):
                    passes[history_idx] = 1
                    # pass at this location cannot have move, skip histroy check
                    continue

                # values < 1 are placed stones, values > 1 are captured stones, equal to 0 are no change
                diff = history[history_idx] - history[history_idx - 1]
                placed_mask = diff < 0
                spacial_data_tensor[0, num_non_history_channels + history_idx][placed_mask] = 1

        # build game data vector
        if is_white:
            num_own_stones = (board_tensor == 2).count_nonzero().float()
            num_opp_stones = (board_tensor == 1).count_nonzero().float()
        else:
            num_own_stones = (board_tensor == 1).count_nonzero().float()
            num_opp_stones = (board_tensor == 2).count_nonzero().float()

        num_empty_locations = (board_tensor == 0).count_nonzero().float()

        you_started = torch.tensor(white_started == is_white, device=device).float()

        # game_data_vector = torch.as_tensor(passes, device=self.device).unsqueeze(0).float()  # shape [1, num_pass]
        board_size = float(self.board_width * self.board_width)
        game_data_vector = torch.cat(
            [
                passes.float(),
                num_own_stones.unsqueeze(0) / board_size,
                num_opp_stones.unsqueeze(0) / board_size,
                num_empty_locations.unsqueeze(0) / board_size,
                you_started.unsqueeze(0),
            ]
        )

        return spacial_data_tensor.to(self.device), game_data_vector.to(self.device)

    def preprocess_state_batch(
        self,
        state_batch: StateBatch,
        history_batch: list[list[State]],
        valid_moves: np.ndarray[Any, np.dtype[np.bool_]],
        is_white_batch_mask: np.ndarray[Any, np.dtype[np.bool_]],  # list[bool]
        white_started: bool,
        device: str,
    ):
        batch_size = state_batch.shape[0]
        num_channels = 7 + self.num_past_steps
        spacial_data_tensor = torch.zeros((batch_size, num_channels, self.board_width, self.board_width), device=device)
        board_tensor_batch = torch.as_tensor(state_batch, device=device)

        # liberty count tensor
        empty_mask = (board_tensor_batch == 0).float().unsqueeze(1)  # Shape [B, 1, H, W]
        stone_mask = (board_tensor_batch == 1) | (board_tensor_batch == 2)

        kernel = self.liberty_kernel_cpu if device == "cpu" else self.liberty_kernel_gpu
        neighbor_liberties = F.conv2d(empty_mask, kernel, padding=1)
        liberty_counts_batch = neighbor_liberties.squeeze(1) * stone_mask  # Shape [B, H, W]

        # ko move tensor
        illegal_moves_tensor_batch = torch.as_tensor(np.invert(valid_moves), device=device)
        # mask black, white and disabled locations so only KO locations are included
        illegal_moves_tensor_batch[
            (board_tensor_batch == 1) | (board_tensor_batch == 2) | (board_tensor_batch == 3)
        ] = 0.0

        spacial_data_tensor[:, 0] = (board_tensor_batch == 3).float()  # disabled
        # if is_white:
        spacial_data_tensor[is_white_batch_mask, 1] = (board_tensor_batch[is_white_batch_mask] == 2).float()  # white
        spacial_data_tensor[is_white_batch_mask, 2] = (board_tensor_batch[is_white_batch_mask] == 1).float()  # black
        # else:
        is_black_batch_mask = ~is_white_batch_mask
        spacial_data_tensor[is_black_batch_mask, 1] = (board_tensor_batch[is_black_batch_mask] == 1).float()  # black
        spacial_data_tensor[is_black_batch_mask, 2] = (board_tensor_batch[is_black_batch_mask] == 2).float()  # white

        spacial_data_tensor[:, 3] = (liberty_counts_batch == 1).float()  # has one liberty
        spacial_data_tensor[:, 4] = (liberty_counts_batch == 2).float()  # has two liberties
        spacial_data_tensor[:, 5] = (liberty_counts_batch == 3).float()  # has three liberties

        spacial_data_tensor[:, 6] = illegal_moves_tensor_batch  # has three liberties

        # process history from most recent to oldest
        passes_batch = torch.zeros((batch_size, self.num_past_steps), dtype=torch.int8, device=device)
        num_non_history_channels = self.policy_net.num_input_channels - self.num_past_steps

        for batch_idx, history in enumerate(history_batch):
            # process t and t-1
            if len(history) > 0:
                if np.array_equal(state_batch[batch_idx], history[0]):
                    passes_batch[batch_idx, 0] = 1
                else:
                    diff = history[0] - state_batch[batch_idx]
                    placed_mask = diff < 0
                    spacial_data_tensor[batch_idx, num_non_history_channels, placed_mask] = 1

                # process history from t-1 to t-k
                for history_idx in range(1, min(self.num_past_steps, len(history))):
                    if np.array_equal(history[history_idx - 1], history[history_idx]):
                        passes_batch[batch_idx, history_idx] = 1
                        # pass at this location cannot have move, skip histroy check
                        continue

                    # values < 1 are placed stones, values > 1 are captured stones, equal to 0 are no change
                    diff = history[history_idx] - history[history_idx - 1]
                    placed_mask = diff < 0
                    spacial_data_tensor[batch_idx, num_non_history_channels + history_idx, placed_mask] = 1

        num_own_stones_batch = torch.zeros(batch_size, device=device)
        num_opp_stones_batch = torch.zeros(batch_size, device=device)

        # build game data vector
        white_counts = (board_tensor_batch == 2).count_nonzero(dim=(1, 2)).float()
        black_counts = (board_tensor_batch == 1).count_nonzero(dim=(1, 2)).float()

        num_own_stones_batch[is_white_batch_mask] = white_counts
        num_opp_stones_batch[is_white_batch_mask] = black_counts
        
        num_own_stones_batch[is_black_batch_mask] = black_counts
        num_opp_stones_batch[is_black_batch_mask] = white_counts

        num_empty_locations = (board_tensor_batch == 0).count_nonzero(dim=(1,2)).float()

        you_started = torch.tensor(white_started == is_white_batch_mask, device=device).float()

        # game_data_vector = torch.as_tensor(passes, device=self.device).unsqueeze(0).float()  # shape [1, num_pass]
        board_size = float(self.board_width * self.board_width)
        game_data_vector = torch.cat(
            [
                passes_batch,
                (num_own_stones_batch / board_size).unsqueeze(1),
                (num_opp_stones_batch / board_size).unsqueeze(1),
                (num_empty_locations / board_size).unsqueeze(1),
                you_started.unsqueeze(1),
            ],
            dim=1,
        )

        return spacial_data_tensor.to(self.device), game_data_vector.to(self.device)

    def deprocess_state(self, state: torch.Tensor, is_white: bool) -> torch.Tensor:
        """
        Deprocess the state tensor into a numpy array.
        """
        new_state = torch.zeros((self.board_height, self.board_height), dtype=torch.int8, device=state.device)
        new_state[state[0] == 1] = 3  # disabled stones
        if is_white:
            new_state[state[1] == 1] = 2  # own stones
            new_state[state[2] == 1] = 1  # opponent stones
        else:
            new_state[state[1] == 1] = 1  # own stones
            new_state[state[2] == 1] = 2  # opponent stones
        return new_state

    @torch.no_grad()  # pyright: ignore
    def predict_eval(
        self,
        uf: UnionFind,
        game_history: list[State],
        valid_moves: np.ndarray[Any, np.dtype[np.bool_]],
        color_is_white: bool,
        white_started: bool,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Given a nodes data, predict the policy and value.

        Args:
            uf (UnionFind)
            game_history (list[State]): History of the entire game, most recent move at index 0.
            valid_moves (np.ndarray[Any, np.dtype[np.bool_]]): Flat numpy array of valid moves, including pass move.
            color_is_white (bool)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        valid_moves_reshaped = valid_moves[:-1].reshape((self.board_width, self.board_height))
        state_tensor, game_data_vector = self.preprocess_state(
            uf.state, game_history, valid_moves_reshaped, color_is_white, white_started, "cpu"
        )

        # Get logits
        policy_logits, outcome, mu, sigma = self.policy_net.forward_mcts_eval(state_tensor, game_data_vector)
        policy_logits_squeezed = policy_logits.squeeze()
        # mask invalid moves and convert to probabilities
        valid_mask = torch.tensor(valid_moves.squeeze(), device=self.device, dtype=torch.bool)
        policy_logits_squeezed[~valid_mask] = -1e9
        final_probs = torch.softmax(policy_logits_squeezed, dim=0)

        return final_probs, outcome.squeeze().item(), mu.item(), sigma.item()

    @torch.no_grad()  # pyright: ignore
    def predict_eval_batch(self, node_batch: list[Node]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(node_batch)
        valid_moves_batch = np.zeros((batch_size, self.board_width, self.board_width), dtype=np.bool_)
        state_batch = np.zeros((batch_size, self.board_width, self.board_width), dtype=np.int8)
        game_history_batch:list[list[State]] = []
        is_white_batch = np.empty(batch_size, dtype=np.bool_)
        
        server_history = node_batch[0].server.go.get_history()
        white_started = node_batch[0].server.go.white_starts

        # build node batch from nodes
        for i, node in enumerate(node_batch):
            valid_moves = node.get_valid_moves()
            valid_moves_batch[i] = valid_moves[:-1].reshape((self.board_width, self.board_width))
            state_batch[i] = node.uf.state
            node_history = node.get_history_ref()
            node_history.extend(server_history)
            game_history_batch.append(node_history)
            is_white_batch[i] = node.is_white

        # valid_moves_reshaped = valid_moves[:-1].reshape((self.board_width, self.board_height))
        state_tensor, game_data_vector = self.preprocess_state_batch(
            state_batch, game_history_batch, valid_moves_batch, is_white_batch, white_started, "cpu"
        )

        # Get logits
        policy_logits, outcome, mu, sigma = self.policy_net.forward_mcts_eval(state_tensor, game_data_vector)
        policy_logits_squeezed = policy_logits.squeeze()
        # mask invalid moves and convert to probabilities
        valid_mask = torch.tensor(valid_moves_batch, device=self.device, dtype=torch.bool)
        policy_logits_squeezed[~valid_mask] = -1e9
        final_probs = torch.softmax(policy_logits_squeezed, dim=1)

        return final_probs, outcome.squeeze(), mu, sigma

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

        # 1. sample batch from train buffer
        batch = self.train_buffer.sample(self.batch_size)

        # tuple[torch.Tensor, torch.Tensor, int, bool, torch.Tensor]
        (pi_list, pi_opp_list, z_list, is_white_list, group_list, score_list, full_search_list, white_started_list) = (
            zip(*batch)
        )

        # 2. Convert data into tensors
        state_tensor_list: list[torch.Tensor] = []
        game_info_vec_list: list[torch.Tensor] = []
        ownership_list: list[torch.Tensor] = []
        for i in range(len(batch)):
            current_group = group_list[i]
            uf = UnionFind.get_uf_from_state(current_group[:, :, GroupIdx.STATE].cpu().numpy(), None)
            valid_moves = current_group[:, :, GroupIdx.VALID_MOVES].cpu().numpy()
            history = current_group[:, :, GroupIdx.HISTORY :]
            history_list = [history[:, :, j].cpu().numpy() for j in range(self.history_length)]

            valid_moves = np.array(valid_moves, dtype=np.bool_)
            t, v = self.preprocess_state(
                uf.state, history_list, valid_moves, is_white_list[i], white_started_list[i], device="cuda"
            )
            state_tensor_list.append(t)
            game_info_vec_list.append(v)
            ownership_list.append(current_group[:, :, 1])

        state_batch = torch.cat(state_tensor_list)  # shape [B, channels, W, H]
        game_info_vec_batch = torch.stack(game_info_vec_list, dim=0)  # shape [B, num past steps + 4]
        pi_batch = torch.stack(pi_list, dim=0).to(device=self.device)  # shape [B, 26]
        pi_opp_batch = torch.stack(pi_opp_list, dim=0).to(device=self.device)  # shape [B, 26]
        z_batch = torch.tensor(z_list, dtype=torch.float, device=self.device)  # [B]
        # z_batch_transformed = torch.stack([(z_batch + 1) / 2, (1 - z_batch) / 2], dim=1)  # [B, 2]
        # z_class_indices = ((1 - z_batch) / 2).long()

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

        full_search_mask = torch.tensor(full_search_list, dtype=torch.bool, device=self.device)  # shape [B]
        full_search_mask_float = full_search_mask.float()

        # 3. feed states trough NN
        # logits shape [B, 26], values shape [B, 1] (assuming your net does that)
        logits_own, logits_opp, outcome_logits, ownership_logits, score_logits, v_pooled = self.policy_net(
            state_batch, game_info_vec_batch
        )

        # 4. Calculate losses
        policy_log_probs_own = F.log_softmax(logits_own, dim=1)  # shape [B, num_actions]
        policy_log_probs_opp = F.log_softmax(logits_opp, dim=1)  # shape [B, num_actions]
        # z_hat = F.log_softmax(outcome_logits[:, :2], dim=1)  # shape [B, 2]
        score_log_probs = F.log_softmax(score_logits, dim=1)  # shape [B, num_possible_scores]
        score_probs = F.softmax(score_logits, dim=1)  # shape [B, num_possible_scores]
        z_hat = outcome_logits[:, 0]
        mu_hat = outcome_logits[:, 1]
        sigma_hat = outcome_logits[:, 2]

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
        score_onehot = torch.zeros((self.batch_size, self.num_possible_scores), device=self.device)
        for i in range(self.batch_size):
            score_idx = self.num_possible_scores // 2 + torch.floor(score_batch[i])
            score_onehot[i, int(score_idx)] = 1.0

        # calculate losses
        # pi_batch is shape [B, num_actions]
        num_full_search_samples = torch.clamp(full_search_mask_float.sum(), min=1.0)  # Avoid division by zero

        raw_policy_loss_own = -(pi_batch * policy_log_probs_own).sum(dim=1)  # shape [B]
        raw_policy_loss_opp = -(pi_opp_batch * policy_log_probs_opp).sum(dim=1)  # shape [B]

        policy_loss_own = (raw_policy_loss_own * full_search_mask).sum() / num_full_search_samples
        policy_loss_opp = ((raw_policy_loss_opp * full_search_mask).sum() / num_full_search_samples) * 0.15

        # game_outcome_value_loss = -(z_batch_transformed * z_hat).sum(dim=1).mean() * 1.5
        # game_outcome_value_loss = F.cross_entropy(outcome_logits[:, :2], z_class_indices) * 1.5
        game_outcome_value_loss = F.mse_loss(z_hat, z_batch) * 1.5

        ownership_loss_weight = 1.5 / (self.board_width * self.board_width)
        ownership_loss = -(o_target_flat * torch.log(ownership_hat_flat)).sum(dim=1).mean() * ownership_loss_weight

        score_pdf_loss = -(score_onehot * score_log_probs).sum(dim=1).mean() * 0.02

        target_cdf = torch.cumsum(score_onehot, dim=1)  # shape [B, num_possible_scores]
        predicted_cdf = torch.cumsum(score_probs, dim=1)  # shape [B, num_possible_scores]
        score_cdf_loss = torch.mean(torch.sum((target_cdf - predicted_cdf) ** 2, dim=1)) * 0.02

        mu_s = torch.sum(self.possible_scores * score_probs, dim=1, keepdim=True)  # shape [B, 1]
        variance_s = torch.sum(((self.possible_scores - mu_s) ** 2) * score_probs, dim=1, keepdim=True)  # shape [B, 1]
        sigma_s = torch.sqrt(variance_s + epsilon).squeeze(1)  # epsilon in case variance_s is 0

        score_mean_loss = F.huber_loss(mu_hat, mu_s.squeeze(1), delta=10.0) * 0.004
        score_std_loss = F.huber_loss(sigma_hat, sigma_s, delta=10.0) * 0.004

        gamma_intermediate = F.relu(self.policy_net.score_head.scale_fc1(v_pooled))
        gamma = self.policy_net.score_head.scale_fc2(gamma_intermediate)
        score_scaling_penalty = 0.0005 * (gamma**2).mean()
        # Assert all loss components are numerical (not NaN)

        assert not torch.isnan(policy_loss_own), "policy_loss_own is NaN"
        assert not torch.isnan(policy_loss_opp), "policy_loss_opp is NaN"
        assert not torch.isnan(game_outcome_value_loss), "game_outcome_value_loss is NaN"
        assert not torch.isnan(ownership_loss), "ownership_loss is NaN"
        assert not torch.isnan(score_pdf_loss), "score_pdf_loss is NaN"
        assert not torch.isnan(score_cdf_loss), "score_cdf_loss is NaN"
        assert not torch.isnan(score_mean_loss), "score_mean_loss is NaN"
        assert not torch.isnan(score_std_loss), "score_std_loss is NaN"
        assert not torch.isnan(score_scaling_penalty), "score_scaling_penalty is NaN"

        loss = (
            policy_loss_own
            + policy_loss_opp
            + game_outcome_value_loss
            + ownership_loss
            + score_pdf_loss
            + score_cdf_loss
            + score_mean_loss
            + score_std_loss
            + score_scaling_penalty
        )

        # 5. Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore
        self.scheduler.step()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        if total_grad_norm > 1.0:
            print(f"Gradient was clipped! Norm before clipping: {total_grad_norm:.4f}")

        current_lr = self.scheduler.get_last_lr()[0]

        if self.plotter is not None:
            if policy_loss_opp != 0:
                self.plotter.update_stat("losses/loss", loss.item())
                self.plotter.update_stat("losses/policy_loss", policy_loss_own.item())
                self.plotter.update_stat("losses/value_loss", game_outcome_value_loss.item())
            self.plotter.update_stat("losses/policy_loss_opp", policy_loss_opp.item())
            self.plotter.update_stat("losses/ownership_loss", ownership_loss.item())
            self.plotter.update_stat("losses/score_pdf_loss", score_pdf_loss.item())
            self.plotter.update_stat("losses/score_cdf_loss", score_cdf_loss.item())
            self.plotter.update_stat("losses/score_mean_loss", score_mean_loss.item())
            self.plotter.update_stat("losses/score_std_loss", score_std_loss.item())
            self.plotter.update_stat("misc/grad_norm", total_grad_norm.item())

        print(
            f"Training step: loss={loss}, policy_loss={policy_loss_own}, value_loss={game_outcome_value_loss}, lr={current_lr}"
        )

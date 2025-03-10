import os
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from plotter import Plotter

State = np.ndarray[Any, np.dtype[np.int8]]


class ResNet(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        num_res_blocks: int = 4,
        num_hiden: int = 64,
        num_past_steps: int = 2,
    ) -> None:
        super().__init__()  # pyright: ignore
        self.initialConvBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=1  # disabled nodes
                + 1  # side (1...black, 0...white)
                + 2  # current black/white
                + num_past_steps * 2,  # past moves
                out_channels=num_hiden,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_hiden),
            nn.ReLU(),
        )

        self.feature_extractor = nn.ModuleList(
            [ResBlock(num_hiden) for _ in range(num_res_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hiden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_width * board_height, board_width * board_height + 1),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hiden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * board_width * board_height, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.initialConvBlock(x)
        for block in self.feature_extractor:
            x = block(x)
        return (self.policyHead(x), self.valueHead(x))


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
    def __init__(self, capacity: int = 10000):
        self.buffer: deque[tuple[State, torch.Tensor, int, list[State], bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: State,
        pi_mcts: torch.Tensor,
        outcome: int,
        history: list[State],
        was_white: bool,
    ):
        self.buffer.append((state, pi_mcts, outcome, history, was_white))

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
        batch_size: int = 64,
        num_past_steps: int = 2,
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
            board_width, board_width, 6, num_past_steps=num_past_steps, num_hiden=96
        ).to(self.device)
        self.policy_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=lr, weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-5
        )

        self.train_buffer = TrainingBuffer()

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
        print(f"Checkpoint saved to {checkpoint_path}")
        self._manage_checkpoints()

    def load_checkpoint(self, filename: str):
        """Loads the model and optimizer state from a file."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path)  # pyright: ignore
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
        state: State,
        pi_mcts: torch.Tensor,
        outcome: int,
        history: list[State],
        was_white: bool,
    ):
        """
        creates 7 augmented versions of the provided state, and pushes all 8 versions into the training buffer.
        """

        self.train_buffer.push(state, pi_mcts, outcome, history, was_white)

        pi_board = pi_mcts[: self.board_width * self.board_height].reshape(
            self.board_width, self.board_height
        )
        pi_pass = pi_mcts[-1]

        # rotate state
        for rot in range(1, 4):
            rotated_state = np.rot90(state, rot)
            rotated_history = [np.rot90(h, rot) for h in history]

            rotated_pi_board = torch.rot90(pi_board, rot)
            rotated_pi = torch.cat([rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])

            self.train_buffer.push(
                rotated_state, rotated_pi, outcome, rotated_history, was_white
            )

        # mirror state
        mirrored_state = np.fliplr(state)
        mirrored_history = [np.fliplr(h) for h in history]
        mirrored_pi_board = torch.fliplr(pi_board)
        mirrored_pi = torch.cat([mirrored_pi_board.flatten(), pi_pass.unsqueeze(0)])

        self.train_buffer.push(
            mirrored_state, mirrored_pi, outcome, mirrored_history, was_white
        )

        for rot in range(1, 4):
            mirrored_rotated_state = np.rot90(mirrored_state, rot)
            mirrored_rotated_history = [np.rot90(h, rot) for h in mirrored_history]

            mirrored_rotated_pi_board = torch.rot90(mirrored_pi_board, rot)
            mirrored_rotated_pi = torch.cat(
                [mirrored_rotated_pi_board.flatten(), pi_pass.unsqueeze(0)]
            )

            self.train_buffer.push(
                mirrored_rotated_state,
                mirrored_rotated_pi,
                outcome,
                mirrored_rotated_history,
                was_white,
            )

    def preprocess_state(
        self, board_state: State, history: list[State], is_white: bool
    ):
        """
        Convert the board (numpy array) into a float tensor of shape [1,8,w,h].
        1 -> 1.0  Black, Channel 1, 3, 5
        2 -> 1.0  White, Channel 2, 4, 6
        3 -> 1.0  Channel 0
        0 -> 0.0  (Empty)
        """
        w, h = self.board_width, self.board_height

        channels: list[torch.Tensor] = []

        # disabled channel
        disabled_channel = torch.zeros(w, h, device=self.device)
        side_channel = (
            torch.ones(w, h, device=self.device)
            if is_white
            else torch.zeros(w, h, device=self.device)
        )
        current_black = torch.zeros(w, h, device=self.device)
        current_white = torch.zeros(w, h, device=self.device)

        mask_black = torch.as_tensor(board_state == 1, device=self.device)
        mask_white = torch.as_tensor(board_state == 2, device=self.device)
        mask_disabled = torch.as_tensor(board_state == 3, device=self.device)
        current_black[mask_black] = 1.0
        current_white[mask_white] = 1.0
        disabled_channel[mask_disabled] = 1.0

        channels.extend([disabled_channel, side_channel, current_black, current_white])

        # parse history and append to channels
        for past_idx in range(self.num_past_steps):

            history_black = torch.zeros(w, h, device=self.device)
            history_white = torch.zeros(w, h, device=self.device)
            if len(history) > past_idx:
                past_step = history[past_idx]

                mask_black = torch.as_tensor(past_step == 1, device=self.device)
                mask_white = torch.as_tensor(past_step == 2, device=self.device)
                history_black[mask_black] = 1.0
                history_white[mask_white] = 1.0

            channels.extend([history_black, history_white])

        board_tensor = torch.stack(channels).unsqueeze(0)

        return board_tensor

    @torch.no_grad()  # pyright: ignore
    def get_actions_eval(
        self,
        board: State,
        valid_moves: np.ndarray[Any, np.dtype[np.bool_]],
        game_history: list[State],
        color_is_white: bool,
    ) -> tuple[torch.Tensor, int]:
        """
        Select action deterministically for evaluation (no exploration).
        """
        state_tensor = self.preprocess_state(board, game_history, color_is_white)

        # Get Q-values
        policy, value = self.policy_net(state_tensor)

        # Select the action with highest Q-value
        return policy.squeeze(0), value.item()

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

        (states_list, pi_list, z_list, history_list, is_white_list) = zip(*batch)

        # 2. Convert Python data into PyTorch tensors
        state_tensor_list: list[torch.Tensor] = []
        for i in range(len(batch)):
            t = self.preprocess_state(states_list[i], history_list[i], is_white_list[i])
            state_tensor_list.append(t)

        state_batch = torch.cat(state_tensor_list)  # shape [B, channels, W, H]
        pi_batch = torch.stack(pi_list, dim=0).to(device=self.device)  # shape [B, 26]
        z_batch = torch.tensor(z_list, dtype=torch.float, device=self.device)  # [B]

        # 3. feed states trough NN
        # logits shape [B, 26], values shape [B, 1] (assuming your net does that)
        logits, values = self.policy_net(state_batch)

        # 4. Calculate losses
        policy_log_probs = F.log_softmax(logits, dim=1)  # shape [B, num_actions]
        # pi_batch is shape [B, num_actions]
        policy_loss = -(pi_batch * policy_log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values.squeeze(), z_batch)

        loss = policy_loss + value_loss

        self.plotter.update_loss(loss.item())
        self.plotter.update_policy_loss(policy_loss.item())
        self.plotter.update_value_loss(value_loss.item())

        # 5. Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore
        self.scheduler.step()

        current_lr = self.scheduler.get_last_lr()[0]
        print(
            f"Training step: loss={loss}, policy_loss={policy_loss}, value_loss={value_loss}, lr={current_lr}"
        )

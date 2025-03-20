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
        num_res_blocks: int = 2,
        num_hiden: int = 32,
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

        self.feature_extractor = nn.ModuleList([ResBlock(num_hiden) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hiden, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * board_width * board_height, board_width * board_height + 1),
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
        self.buffer: deque[tuple[State, torch.Tensor, int, list[State], bool]] = deque(maxlen=capacity)

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
        lr: float = 8e-5,
        batch_size: int = 128,
        num_past_steps: int = 3,
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

        self.policy_net = ResNet(board_width, board_width, 2, num_past_steps=num_past_steps).to(self.device)
        self.policy_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=wheight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=5e-6)

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

        pi_board = pi_mcts[: self.board_width * self.board_height].reshape(self.board_width, self.board_height)
        pi_pass = pi_mcts[-1]

        # rotate state
        for rot in range(1, 4):
            rotated_state = np.rot90(state, rot).copy()
            rotated_history = [np.rot90(h, rot).copy() for h in history]

            rotated_pi_board = torch.rot90(pi_board, rot)
            rotated_pi = torch.cat([rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])

            self.train_buffer.push(rotated_state, rotated_pi, outcome, rotated_history, was_white)

        # mirror state
        mirrored_state = np.fliplr(state).copy()
        mirrored_history = [np.fliplr(h).copy() for h in history]
        mirrored_pi_board = torch.fliplr(pi_board)
        mirrored_pi = torch.cat([mirrored_pi_board.flatten(), pi_pass.unsqueeze(0)])

        self.train_buffer.push(mirrored_state, mirrored_pi, outcome, mirrored_history, was_white)

        for rot in range(1, 4):
            mirrored_rotated_state = np.rot90(mirrored_state, rot).copy()
            mirrored_rotated_history = [np.rot90(h, rot).copy() for h in mirrored_history]

            mirrored_rotated_pi_board = torch.rot90(mirrored_pi_board, rot)
            mirrored_rotated_pi = torch.cat([mirrored_rotated_pi_board.flatten(), pi_pass.unsqueeze(0)])

            self.train_buffer.push(
                mirrored_rotated_state,
                mirrored_rotated_pi,
                outcome,
                mirrored_rotated_history,
                was_white,
            )

    def preprocess_state(self, state: State, history: list[State], is_white: bool):
        """
        Convert the board (numpy array) into a float tensor of shape [1,8,w,h].
        1 -> 1.0  Black, Channel 1, 3, 5
        2 -> 1.0  White, Channel 2, 4, 6
        3 -> 1.0  Channel 0
        0 -> 0.0  (Empty)
        """
        num_channels = 4 + 2 * self.num_past_steps
        result = torch.zeros((1, num_channels, self.board_width, self.board_width), device=self.device)
        board_tensor = torch.as_tensor(state, device=self.device)

        result[0, 0] = (board_tensor == 3).float()  # disabled
        result[0, 1].fill_(is_white)  # side
        result[0, 2] = (board_tensor == 1).float()  # black
        result[0, 3] = (board_tensor == 2).float()  # white

        # process history
        for past_idx in range(min(self.num_past_steps, len(history))):
            past_step = torch.as_tensor(history[past_idx], device=self.device)
            result[0, 4 + past_idx * 2] = (past_step == 1).float()  # black
            result[0, 5 + past_idx * 2] = (past_step == 2).float()  # white

        return result

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

        # Get logits
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
            # side_str = "White" if is_white_list[i] else "Black"
            # outcome_str = "Win" if z_list[i] > 0 else "Loss"
            # print(f"[DEBUG] Next to move: {side_str}, Label says: {outcome_str} (z={z_list[i]})")
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
        print(f"Training step: loss={loss}, policy_loss={policy_loss}, value_loss={value_loss}, lr={current_lr}")

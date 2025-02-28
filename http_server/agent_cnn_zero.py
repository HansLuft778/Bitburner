import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from plotter import Plotter


class ResNet(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        num_res_blocks=4,
        num_hiden=64,
        num_past_steps=2,
    ):
        super().__init__()
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
            [ResBlock(num_hiden) for i in range(num_res_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hiden, 32, kernel_size=3, padding=1),
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

    def forward(self, x):
        x = self.initialConvBlock(x)
        for block in self.feature_extractor:
            x = block(x)
        return (self.policyHead(x), self.valueHead(x))


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class TrainingBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: list[str],
        pi_mcts: torch.Tensor,
        outcome: int,
        history: list[list[str]],
        was_white: bool,
    ):
        self.buffer.append((state, pi_mcts, outcome, history, was_white))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class AlphaZeroAgent:
    def __init__(
        self,
        board_width: int,
        plotter: Plotter,
        lr=1e-4,
        batch_size=64,
        num_past_steps=2,
    ):
        self.board_width = board_width
        self.board_height = board_width
        self.plotter = plotter
        self.batch_size = batch_size
        self.num_past_steps = num_past_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ResNet(
            board_width, board_width, 4, num_past_steps=num_past_steps
        ).to(self.device)
        # self.policy_net.load_state_dict(torch.load("mcts_zero_works_2.pt"))
        self.policy_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.train_buffer = TrainingBuffer()

    def preprocess_state(
        self, board_state: np.ndarray, history: list[np.ndarray], is_white: bool
    ):
        """
        Convert the board (numpy array) into a float tensor of shape [1,8,w,h].
        1 -> 1.0  Black, Channel 1, 3, 5
        2 -> 1.0  White, Channel 2, 4, 6
        3 -> 1.0  Channel 0
        0 -> 0.0  (Empty)
        """
        w, h = self.board_width, self.board_height

        channels = []

        # disabled channel
        disabled_channel = torch.zeros(w, h, device=self.device)
        side_channel = (
            torch.ones(w, h, device=self.device)
            if is_white
            else torch.zeros(w, h, device=self.device)
        )
        current_black = torch.zeros(w, h, device=self.device)
        current_white = torch.zeros(w, h, device=self.device)

        # mask_black = torch.from_numpy(board_state == 1).to(self.device)
        # mask_white = torch.from_numpy(board_state == 2).to(self.device)
        # mask_disabled = torch.from_numpy(board_state == 3).to(self.device)
        # current_black[mask_black] = 1.0
        # current_white[mask_white] = 1.0
        # disabled_channel[mask_disabled] = 1.0

        for x in range(w):
            for y in range(h):
                ch = board_state[x][y]
                if ch == 1:
                    current_black[x][y] = 1.0
                elif ch == 2:
                    current_white[x][y] = 1.0
                elif ch == 3:
                    disabled_channel[x][y] = 1.0

        channels.extend([disabled_channel, side_channel, current_black, current_white])

        # parse history and append to channels
        for past_idx in range(self.num_past_steps):

            history_black = torch.zeros(w, h, device=self.device)
            history_white = torch.zeros(w, h, device=self.device)
            if len(history) > past_idx:
                past_step = history[past_idx]
                for x in range(w):
                    for y in range(h):
                        ch = past_step[x][y]
                        if ch == 1:
                            history_black[x][y] = 1.0
                        elif ch == 2:
                            history_white[x][y] = 1.0

            channels.extend([history_black, history_white])

        board_tensor = torch.stack(channels).unsqueeze(0)

        return board_tensor

    @torch.no_grad()
    def get_actions_eval(
        self, board, valid_moves, game_history, color_is_white
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

        # 5. Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(
            f"Training step: loss={loss}, policy_loss={policy_loss}, value_loss={value_loss}"
        )

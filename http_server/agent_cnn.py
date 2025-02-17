import os
import random
from collections import deque
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from plotter import Plotter


class GoCNN(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        num_channels=32,
        hidden_size=128,
        num_past_steps=2,
    ):
        super(GoCNN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(
            in_channels=1
            + 1
            + 2
            + num_past_steps * 2,  # disabled, current black/while, past moves
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )

        self.fc_input_size = num_channels * board_width * board_height
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, board_width * board_height + 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, self.fc_input_size)

        x = F.relu(self.fc1(x))
        q_values = self.fc_out(x)
        return q_values


# class GoCNN(nn.Module):
#     def __init__(
#         self,
#         board_width: int,
#         board_height: int,
#         num_channels=128,
#         hidden_size=512,
#         num_past_steps=2,
#     ):
#         super(GoCNN, self).__init__()

#         self.board_width = board_width
#         self.board_height = board_height

#         # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
#         self.conv1 = nn.Conv2d(
#             in_channels=1  # disabled nodes
#             + 1  # side (1...black, 0...white)
#             + 2  # current black/white
#             + num_past_steps * 2,  # past moves
#             out_channels=num_channels,  # 32
#             kernel_size=3,
#             padding=1,
#             # padding_mode="same",
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=num_channels,
#             out_channels=num_channels,  # 64
#             kernel_size=3,
#             padding=1,
#             # padding_mode="same",
#         )
#         self.conv3 = nn.Conv2d(
#             in_channels=num_channels,
#             out_channels=num_channels,  # 128
#             kernel_size=3,
#             padding=1,
#             # padding_mode="same",
#         )
#         self.conv4 = nn.Conv2d(
#             in_channels=num_channels,
#             out_channels=num_channels,  # 256
#             kernel_size=3,
#             padding=1,
#             # padding_mode="same",
#         )

#         self.fc_input_size = num_channels * board_width * board_height
#         self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
#         self.fc_out = nn.Linear(hidden_size, board_width * board_height + 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))

#         x = x.view(-1, self.fc_input_size)

#         x = F.relu(self.fc1(x))
#         q_values = self.fc_out(x)
#         return q_values


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        legal_mask: list[bool],
    ):
        """
        state: The current state (the board, in your case) before we take an action.
        action: The action the agent chose at that state (e.g. “place router at (x,y)” or “pass”).
        reward (float): The immediate reward we got from taking that action (+1/-1 at game end, 0 for moves in between)
        next_state: The state of the board after the action has been applied.
        done (bool): A boolean indicating if the episode/game has ended.
        legal_mask: A map to mask out illegal moves
        """
        self.buffer.append((state, action, reward, next_state, done, legal_mask))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def getCheckpointFile():
    files = os.listdir("models/checkpoints/")
    newest_file = None
    newest_time = None
    for file in files:
        if file.startswith("checkpoint_") and file.endswith(".pt"):
            dt_str = file.replace("checkpoint_", "").replace(".pt", "")
            # Convert '2025-01-21T17-08-19' -> '2025-01-21T17:08:19'
            date_part, time_part = dt_str.split("T")
            time_part = time_part.replace("-", ":")
            final_dt_str = date_part + "T" + time_part
            dt = datetime.datetime.fromisoformat(final_dt_str)
            if newest_time is None or dt > newest_time:
                newest_time = dt
                newest_file = file
    return newest_file


class DQNAgentCNN:
    def __init__(
        self,
        board_width: int,
        board_height: int,
        plotter: Plotter | None,
        lr=1e-4,
        gamma=0.95,
        batch_size=64,
        num_past_steps=2,
    ):
        self.board_width = board_width
        self.board_height = board_height
        self.plotter = plotter
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_past_steps = num_past_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check if there is a model to laod

        self.policy_net = GoCNN(
            board_width, board_height, num_past_steps=num_past_steps
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Exploration parameter
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        checkpoint_file = getCheckpointFile()
        if checkpoint_file != "" and checkpoint_file is not None:
            # if os.path.isfile("models/model_cnn.pt"):
            print(f"loading checkpoint {checkpoint_file}")
            checkpoint = torch.load(f"models/checkpoints/{checkpoint_file}")
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.policy_net.train()
            self.epsilon = checkpoint["epsilon"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # if os.path.isfile("models/model_cnn.pt"):
        #     self.policy_net.load_state_dict(torch.load("models/model_cnn.pt"))

        self.target_net = GoCNN(board_width, board_height).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.replay_buffer = ReplayBuffer()

    def preprocess_state(
        self, board_state: list[str], history: list[list[str]], is_white=False
    ):
        """
        Convert the board (list of strings, e.g. ["XX.O#", ...]) into a float tensor of shape [1,7,w,h].
        '#' -> 1.0  Channel 0
        'X' -> 1.0  Black, Channel 1, 3, 5
        'O' -> 1.0  White, Channel 2, 4, 6
        '.' -> 0.0  (Empty)
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

        for x in range(w):
            for y in range(h):
                ch = board_state[x][y]
                if ch == "X":
                    current_black[x][y] = 1.0
                elif ch == "O":
                    current_white[x][y] = 1.0
                elif ch == "#":
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
                        if ch == "X":
                            history_black[x][y] = 1.0
                        elif ch == "O":
                            history_white[x][y] = 1.0

            channels.extend([history_black, history_white])

        board_tensor = torch.stack(channels).unsqueeze(0)

        return board_tensor

    def select_action(
        self, board_state: list[str], legal_moves: list[bool], history: list[list[str]]
    ) -> int:
        """
        Epsilon-greedy action selection:
         - with probability epsilon, pick a random valid move
         - otherwise pick move with max Q-value
        We also include one extra action index for "pass".
        """
        state_tensor = self.preprocess_state(board_state, history)

        # board_size = self.board_width * self.board_height
        if random.random() < self.epsilon:
            # Random move
            # For simplicity, choose from [0..board_size] uniformly
            # You might want to only choose from valid moves
            print(f"epsilon {self.epsilon}")
            legal_idx = [i for i, bit in enumerate(legal_moves) if bit]
            action_idx = random.choice(legal_idx)
            # action_idx = random.randint(0, board_size)  # last index for "pass"
        else:
            print("-------Q-------")
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)  # shape [1, board_size+1]
                legal_mask_tensor = torch.tensor(legal_moves, device=self.device)

                # create mask with illegal moves not being picked
                masked_q_values = q_values.clone()
                masked_q_values[0, ~legal_mask_tensor] = (
                    -1e9  # Set illegal moves to negative
                )

                # print(f"Original q_values: {q_values}")
                # print(f"Masked q_values: {masked_q_values}")

                action_idx = masked_q_values.argmax(dim=1).item()
                # print(f"action index: {action_idx}")
                # print(f"board: {board_state}")
                # print(f"board tensor: {state_tensor}")
        return action_idx

    def select_action_eval(
        self,
        board: list[str],
        valid_moves: list[bool],
        game_history: list[list[str]],
        color_is_white: bool,
    ):
        """
        Select action deterministically for evaluation (no exploration).
        """
        state = self.preprocess_state(board, game_history, color_is_white)

        # Get Q-values
        q_values = self.policy_net(state)

        # Mask invalid moves with large negative number
        mask = torch.tensor(valid_moves, dtype=torch.float32, device=self.device)
        q_values = q_values + (mask - 1) * 1e9

        # Select the action with highest Q-value
        return torch.argmax(q_values).item()

    def decode_action(self, action_idx: int):
        """
        Convert the action index back to (x, y) or 'pass'.
        If action_idx == board_size, then 'pass'.
        """
        board_size = self.board_width * self.board_height
        if action_idx == board_size:
            return "pass"
        else:
            x = action_idx // self.board_height
            y = action_idx % self.board_height
            return (x, y)

    def train_step(self):
        """
        Sample from replay buffer, do a DQN update.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        print(
            "==========================================================================="
        )
        # 2. Sample a batch of transitions
        batch = self.replay_buffer.sample(self.batch_size)
        # print(f"batch: {batch}")

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            legal_batch,
        ) = zip(*batch)

        # 3. Convert Python data into PyTorch tensors
        state_batch = torch.cat(state_batch)  # shape [batch_size, board_size]
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float, device=self.device)

        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)

        next_state_batch = torch.cat(next_state_batch)  # shape [batch_size, board_size]

        # 4. Calculate current Q-values for each (state, action)
        # q_values will be shape [batch_size, num_actions]
        q_values = self.policy_net(state_batch)
        # print(f"q-values: {q_values}")
        # gather(1, action_batch) picks the Q-value for the specific action each transition took
        # after gather, q_values has shape [batch_size, 1]
        q_values = q_values.gather(1, action_batch.view(-1, 1))  # shape [batch_size, 1]
        # print(f"gathered q-values: {q_values}")

        # 5. Calculate target Q-values using the target net
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            for i in range(self.batch_size):
                # Convert the i-th legality array into a tensor, then set invalid to -inf
                next_legal_mask_tensor = torch.tensor(
                    legal_batch[i], device=self.device, dtype=torch.bool
                )
                next_q_values[i, ~next_legal_mask_tensor] = -1e9
                # print(f"next q values {i}: {next_q_values[i]}")

            # max_next_q_values = self.target_net(next_state_batch).max(dim=1)[0]
            max_next_q_values = next_q_values.max(dim=1)[0]
            # print(f"max next q: {max_next_q_values}")
            # DQN target
            target_q = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
            # print(f"target_q: {target_q}")

        # 6. Compute loss between current Q-values and target Q-values
        # q_values is [batch_size, 1], target_q is [batch_size], so we squeeze q_values
        loss = F.mse_loss(q_values.squeeze(), target_q)
        self.plotter.update_loss(loss)
        print(f"LOSS: {loss}")

        # 7. Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

    def update_target_network(self):
        """
        Periodically copy policy_net weights to target_net
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """
        Gradually reduce epsilon for less random exploration over time.
        """
        self.plotter.update_epsilon(self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_propabilities_from_state(self, state: list[str], history: list[list[str]]):
        tensor = self.preprocess_state(state, history)
        return self.policy_net(tensor)

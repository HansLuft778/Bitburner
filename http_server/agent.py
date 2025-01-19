import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from plotter import Plotter


class GoNet(nn.Module):
    def __init__(self, board_width: int, board_height: int, hidden_size=128):
        super(GoNet, self).__init__()
        # For demonstration, flatten the board and pass it through a few linear layers.
        # For real Go, you'd want a CNN or an advanced architecture.
        self.board_size = board_width * board_height

        self.fc1 = nn.Linear(self.board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output: Q-values for each possible action + 1 for "pass"
        # The total possible moves = board_width * board_height + 1
        self.fc_out = nn.Linear(hidden_size, self.board_size + 1)

    def forward(self, x):
        # x shape: (batch_size, board_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc_out(x)
        return q_values


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
        legal_mask: list[bool],
    ):
        """
        state: The current state (the board, in your case) before we take an action. Often a PyTorch tensor.
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


class DQNAgent:
    def __init__(
        self,
        board_width: int,
        board_height: int,
        plotter: Plotter,
        lr=3e-4,
        gamma=0.95,
        batch_size=64,
    ):
        self.board_width = board_width
        self.board_height = board_height
        self.plotter = plotter
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check if there is a model to laod

        self.policy_net = GoNet(board_width, board_height).to(self.device)

        if os.path.isfile("models/model.pr"):
            self.policy_net.load_state_dict(torch.load("models/model.pt"))

        self.target_net = GoNet(board_width, board_height).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer()

        # Exploration parameter
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def preprocess_state(self, board: list[str]):
        """
        Convert the board (list of strings, e.g. ["XX.O.", ...]) into a float tensor.
        For example:
        'X' -> 1.0  (Black)
        'O' -> -1.0 (White)
        '.' -> 0.0  (Empty)
        '#' -> 0.0  (Dead node, treat as non-playable)
        etc.
        """
        w = self.board_width
        h = self.board_height
        flattened = []
        # board[x][y] or board[column_index][row_index]
        # This might depend on your input format, so adjust accordingly.
        # We'll assume "board" is board_width strings, each string is board_height in length.
        for x in range(w):
            for y in range(h):
                ch = board[x][y]
                if ch == "X":
                    flattened.append(1.0)
                elif ch == "O":
                    flattened.append(-1.0)
                else:
                    # '.' or '#'
                    flattened.append(0.0)
        state_tensor = torch.tensor(
            flattened, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        return state_tensor  # shape [1, board_size]

    def select_action(self, board_state: list[str], legal_moves: list[bool]):
        """
        Epsilon-greedy action selection:
         - with probability epsilon, pick a random valid move
         - otherwise pick move with max Q-value
        We also include one extra action index for "pass".
        """
        state_tensor = self.preprocess_state(board_state)

        # board_size = self.board_width * self.board_height
        if random.random() < self.epsilon:
            # Random move
            # For simplicity, choose from [0..board_size] uniformly
            # You might want to only choose from valid moves
            print("epsilon")
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
                    -1e9  # Set illegal moves to negative infinity
                )

                print(f"Original q_values: {q_values}")
                print(f"Masked q_values: {masked_q_values}")

                action_idx = masked_q_values.argmax(dim=1).item()
                print(f"action index: {action_idx}")
                print(f"board: {board_state}")
                print(f"board tensor: {state_tensor}")
        return action_idx

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
        print(f"q-values: {q_values}")
        # gather(1, action_batch) picks the Q-value for the specific action each transition took
        # after gather, q_values has shape [batch_size, 1]
        q_values = q_values.gather(1, action_batch.view(-1, 1))  # shape [batch_size, 1]
        print(f"gathered q-values: {q_values}")

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
            print(f"max next q: {max_next_q_values}")
            # DQN target
            target_q = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
            print(f"target_q: {target_q}")

        # 6. Compute loss between current Q-values and target Q-values
        # q_values is [batch_size, 1], target_q is [batch_size], so we squeeze q_values
        loss = F.mse_loss(q_values.squeeze(), target_q)
        self.plotter.update_loss(loss)
        print(f"LOSS: {loss}")

        # 7. Optimize the policy_net
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
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

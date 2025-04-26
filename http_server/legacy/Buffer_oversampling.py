from Go.Go_uf import UnionFind
import torch
import numpy as np
from typing import Any
from collections import deque
import random
from go_types import State


# UnionFind, bool, torch.Tensor, list[State], torch.Tensor | None
class BufferElement:
    def __init__(
        self,
        uf: UnionFind,
        is_white: bool,
        pi_mcts: torch.Tensor,
        history: list[State],
        valid_moves: np.ndarray[Any, np.dtype[np.bool_]],
        full_search: bool,
        pi_mcts_response: torch.Tensor | None = None,
    ):
        self.uf = uf
        self.is_white = is_white
        self.pi_mcts = pi_mcts
        self.history = history
        self.valid_moves = valid_moves
        self.full_search = full_search
        self.pi_mcts_response = pi_mcts_response


TrainBE = tuple[torch.Tensor, torch.Tensor, int, bool, torch.Tensor, float, bool, bool]


class TrainingBuffer:
    def __init__(self, capacity: int = 75000):
        self.buffer: deque[TrainBE] = deque(maxlen=capacity)

    def push(
        self,
        pi_mcts: torch.Tensor,
        pi_opp: torch.Tensor,
        outcome: int,
        was_white: bool,
        group: torch.Tensor,
        score: float,
        full_search: bool,
        white_started: bool,
    ):
        self.buffer.append((pi_mcts, pi_opp, outcome, was_white, group, score, full_search, white_started))

    def sample(self, batch_size: int, oversample_factor: float = 4.0) -> list[TrainBE]:

        first_player_wins: list[TrainBE] = []
        second_player_wins: list[TrainBE] = []

        for sample in self.buffer:
            outcome = sample[2]
            was_white = sample[3]
            white_started = sample[7]
            is_first_player = was_white == white_started

            if is_first_player:
                if outcome == 1:  # First player made move AND won
                    first_player_wins.append(sample)
                else:  # First player made move AND lost (i.e., second player won)
                    second_player_wins.append(sample)
            else:
                if outcome == 1:  # Second player made move AND won
                    second_player_wins.append(sample)
                else:  # Second player made move AND lost (i.e., first player won)
                    first_player_wins.append(sample)

        # case one list is empty
        if not first_player_wins or not second_player_wins:
            print("Warning: Not enough diversity in outcomes for oversampling. Using random sampling.")
            return random.sample(list(self.buffer), batch_size)

        # Calculate sampling probabilities
        num_first_wins = len(first_player_wins)
        num_second_wins = len(second_player_wins)

        # Assign higher probability weight to second player wins
        prob_second_win = oversample_factor / (num_first_wins * 1.0 + num_second_wins * oversample_factor)
        prob_first_win = 1.0 / (num_first_wins * 1.0 + num_second_wins * oversample_factor)

        combined_samples = first_player_wins + second_player_wins
        probabilities = [prob_first_win] * num_first_wins + [prob_second_win] * num_second_wins
        probabilities = np.array(probabilities, dtype=np.float64)
        probabilities /= probabilities.sum()  # Ensure it sums to 1

        sampled_indices = np.random.choice(len(combined_samples), size=batch_size, p=probabilities)

        batch: list[TrainBE] = [combined_samples[i] for i in sampled_indices]
        return batch

    def __len__(self):
        return len(self.buffer)

from Go.Go_uf import UnionFind
import torch
import numpy as np
from typing import Any
from collections import deque
import random


State = np.ndarray[Any, np.dtype[np.int8]]


# UnionFind, bool, torch.Tensor, list[State], torch.Tensor | None
class BufferElement:
    def __init__(
        self,
        uf: UnionFind,
        is_white: bool,
        pi_mcts: torch.Tensor,
        history: list[State],
        valid_moves: np.ndarray[Any, np.dtype[np.bool_]],
        pi_mcts_response: torch.Tensor | None = None,
    ):
        self.uf = uf
        self.is_white = is_white
        self.pi_mcts = pi_mcts
        self.history = history
        self.valid_moves = valid_moves
        self.pi_mcts_response = pi_mcts_response


class TrainingBuffer:
    def __init__(self, capacity: int = 75000):
        self.buffer_white: deque[tuple[torch.Tensor, torch.Tensor, int, bool, torch.Tensor, float]] = deque(
            maxlen=capacity
        )
        self.buffer_black: deque[tuple[torch.Tensor, torch.Tensor, int, bool, torch.Tensor, float]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        pi_mcts: torch.Tensor,
        pi_opp: torch.Tensor,
        outcome: int,
        was_white: bool,
        group: torch.Tensor,
        score: float,
    ):
        if was_white:
            self.buffer_white.append((pi_mcts, pi_opp, outcome, was_white, group, score))
        else:
            self.buffer_black.append((pi_mcts, pi_opp, outcome, was_white, group, score))

    def sample(self, batch_size: int):
        white_sample = random.sample(self.buffer_white, batch_size // 2)
        black_sample = random.sample(self.buffer_black, batch_size // 2)
        merged_sample = white_sample + black_sample
        random.shuffle(merged_sample)
        return merged_sample

    def __len__(self):
        return min(len(self.buffer_white), len(self.buffer_black)) * 2

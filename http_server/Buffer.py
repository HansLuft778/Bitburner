from Go.Go_uf import UnionFind
import torch
import numpy as np
from typing import Any


State = np.ndarray[Any, np.dtype[np.int8]]


# UnionFind, bool, torch.Tensor, list[State], torch.Tensor | None
class BufferElement:
    def __init__(
        self,
        uf: UnionFind,
        is_white: bool,
        pi_mcts: torch.Tensor,
        history: list[State],
        pi_mcts_response: torch.Tensor | None = None,
    ):
        self.uf = uf
        self.is_white = is_white
        self.pi_mcts = pi_mcts
        self.history = history
        self.pi_mcts_response = pi_mcts_response

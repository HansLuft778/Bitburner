import numpy as np
from typing import Any

State = np.ndarray[Any, np.dtype[np.int8]]


class ZobristHash:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.table = np.array(
            [[np.random.randint(0, 2**64, dtype=np.uint64) for _ in range(2)] for _ in range(board_size * board_size)],
            dtype=np.uint64,
        )

    def compute_hash(self, state: State, player_to_move: int) -> np.uint64:
        """Compute the Zobrist hash for the entire board state"""
        h: np.uint64 = np.uint64(0)
        for i in range(self.board_size):
            for j in range(self.board_size):
                pos = i * self.board_size + j
                if state[i, j] == 1:  # Black stone
                    h ^= self.table[pos][0]
                elif state[i, j] == 2:  # White stone
                    h ^= self.table[pos][1]

        return h

    def update_hash(self, hash: np.uint64, pos: int, old_value: int, new_value: int) -> np.uint64:
        """Update the hash incrementally after a move"""
        # Remove old piece if any
        if old_value == 1:
            hash ^= self.table[pos][0]
        elif old_value == 2:
            hash ^= self.table[pos][1]

        # Add new piece if any
        if new_value == 1:
            hash ^= self.table[pos][0]
        elif new_value == 2:
            hash ^= self.table[pos][1]

        return hash

    def remove_stone(self, hash: np.uint64, pos: int, removed_stone: int) -> np.uint64:
        if removed_stone == 1:
            hash ^= self.table[pos][0]
        elif removed_stone == 2:
            hash ^= self.table[pos][1]

        return hash

    def add_stone(self, hash: np.uint64, pos: int, new_value: int) -> np.uint64:
        if new_value == 1:
            hash ^= self.table[pos][0]
        elif new_value == 2:
            hash ^= self.table[pos][1]

        return hash

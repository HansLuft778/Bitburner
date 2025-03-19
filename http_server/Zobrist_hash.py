import numpy as np
import random
from typing import Any

State = np.ndarray[Any, np.dtype[np.int8]]


class ZobristHash:
    def __init__(self, board_size: int):
        random.seed(0)
        self.board_size = board_size
        self.table = np.array(
            [[random.getrandbits(64) for _ in range(2)] for _ in range(board_size * board_size)], dtype=np.uint64
        )
        self.player_hash = random.getrandbits(64)

    def compute_hash(self, state: State, player_to_move: int) -> int:
        """Compute the Zobrist hash for the entire board state"""
        h = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                pos = i * self.board_size + j
                if state[i, j] == 1:  # Black stone
                    h ^= self.table[pos][0]
                elif state[i, j] == 2:  # White stone
                    h ^= self.table[pos][1]

        # Add player to move in the hash
        if player_to_move == 2:  # If white to move
            h ^= self.player_hash

        return h

    def update_hash(
        self, hash: int, pos: int, old_value: int, new_value: int, player_to_move: int, next_player: int
    ) -> int:
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

        # Update player to move
        if player_to_move != next_player:
            hash ^= self.player_hash

        return hash

    def remove_stone(self, hash: int, pos: int, removed_stone: int) -> int:
        """Update the hash incrementally after a capture"""
        if removed_stone == 1:
            hash ^= self.table[pos][0]
        elif removed_stone == 2:
            hash ^= self.table[pos][1]

        return hash

import random
import numpy as np
from typing import Any

State = np.ndarray[Any, np.dtype[np.int8]]


class GoStateGenerator:
    def __init__(self, width: int) -> None:
        self.board_width = width
        self.sizes = [5, 7, 9, 13, 19]

    def generate_board_state(self) -> State:
        """
        Generates a board state from the current state of the game.
        """
        # Initialize the board state with ones (representing valid points)
        state = np.ones((self.board_width, self.board_width), dtype=np.int8)

        should_remove_corner = random.randint(0, 4) == 0
        should_remove_rows = not should_remove_corner and random.randint(0, 4) == 0
        should_add_center_break = not should_remove_corner and not should_remove_rows and random.randint(0, 3) == 0
        obstacle_type_count = int(should_remove_corner) + int(should_remove_rows) + int(should_add_center_break)

        edge_dead_count = random.randint(1, int((self.sizes.index(self.board_width) + 2 - obstacle_type_count) * 1.5))

        if should_remove_corner:
            state = self.add_dead_corners(state)

        if should_add_center_break:
            state = self.add_center_break(state)

        state = self.randomize_rotation(state)

        if should_remove_rows:
            state = self.remove_rows(state)

        state = self.add_dead_nodes_to_edge(state, edge_dead_count)

        return state

    def add_dead_corners(self, state: State) -> State:
        scale = self.sizes.index(self.board_width) + 1

        state = self.add_dead_corner(state, scale)

        if random.randint(0, 3) == 0:
            state = self.rotate_n_times(state, 2)  # 180 degree rotation
            state = self.add_dead_corner(state, scale - 2)

        return self.randomize_rotation(state)

    def add_dead_corner(self, state: State, scale: int) -> State:
        current_size = scale
        for i in range(scale):
            if i < current_size:
                if random.randint(0, 1) == 1:
                    current_size -= 1
                for j in range(current_size):
                    if state[i][j] != 0:
                        state[i][j] = 0
        return state

    def add_center_break(self, state: State) -> State:
        size = state.shape[0]
        max_offset = self.sizes.index(self.board_width)
        x_index = random.randint(0, max_offset * 2) - max_offset + size // 2
        length = random.randint(1, size // 2 - 1)

        for i in range(length):
            state[x_index][i] = 0

        return self.randomize_rotation(state)

    def rotate_90_degrees(self, state: State) -> State:
        return np.rot90(state, k=3)

    def randomize_rotation(self, state: State) -> State:
        return self.rotate_n_times(state, random.randint(0, 3))

    def rotate_n_times(self, state: State, rotations: int) -> State:
        for _ in range(rotations):
            state = self.rotate_90_degrees(state)
        return state

    def remove_rows(self, state: State) -> State:
        rows_to_remove = max(random.randint(-2, self.sizes.index(self.board_width)), 1)

        rotation_before = random.randint(0, 3)
        state = self.rotate_n_times(state, rotation_before)

        for i in range(rows_to_remove):
            state[i] = np.zeros_like(state[i])

        # Rotate back to original orientation plus additional random rotation
        # to ensure even distribution
        state = self.rotate_n_times(state, (4 - rotation_before) % 4)
        state = self.randomize_rotation(state)

        return state

    def add_dead_nodes_to_edge(self, state: State, max_per_edge: int) -> State:
        size = state.shape[0]
        for _ in range(4):
            count = random.randint(0, max_per_edge)
            for _ in range(count):
                y_index = max(random.randint(-2, size - 1), 0)
                state[0][y_index] = 0
            state = self.rotate_90_degrees(state)
        return state

    def print_board(self, state: State) -> None:
        """Debug method to visualize the board state."""
        symbols = {0: "⬛", 1: "⬜"}  # 0 for removed spaces, 1 for valid spaces
        print("Board visualization:")
        print("  " + " ".join([str(i) for i in range(self.board_width)]))
        for i in range(self.board_width):
            row = [symbols[state[i, j]] for j in range(self.board_width)]
            print(f"{i} {' '.join(row)}")


if __name__ == "__main__":
    generator = GoStateGenerator(5)
    board_state = generator.generate_board_state()
    generator.print_board(board_state)

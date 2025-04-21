import random
import numpy as np
from typing import Any

State = np.ndarray[Any, np.dtype[np.int8]]


class GoStateGenerator:
    def __init__(self, width: int) -> None:
        self.board_width = width
        # Ensure board size is valid if needed, or handle potential IndexError
        self.sizes = [5, 7, 9, 13, 19]
        if width not in self.sizes:
            raise ValueError(f"Unsupported board width: {width}. Supported sizes: {self.sizes}")
        self.scale = self.sizes.index(self.board_width)

    def generate_board_state(self) -> State:
        """
        Generates a board state matching the Bitburner IPvGO Board State Generation
        """
        # Initialize the board state with ones (representing valid points)
        state = np.ones((self.board_width, self.board_width), dtype=np.int8)

        # --- Probabilities ---
        should_remove_corner = random.randint(0, 4) == 0
        should_remove_rows = not should_remove_corner and random.randint(0, 4) == 0
        should_add_center_break = not should_remove_corner and not should_remove_rows and random.randint(0, 3) != 0
        obstacle_type_count = int(should_remove_corner) + int(should_remove_rows) + int(should_add_center_break)

        max_edge_val = int((self.scale + 2 - obstacle_type_count) * 1.5)
        # Ensure max_edge_val is at least 1 for randint range
        edge_dead_count = random.randint(1, max(1, max_edge_val))

        # Obstacle Application Order
        if should_remove_corner:
            state = self.add_dead_corners(state)

        if should_add_center_break:
            state = self.add_center_break(state)

        # Apply random rotation *before* potentially removing rows
        state = self.randomize_rotation(state)

        if should_remove_rows:
            state = self.remove_rows(state)

        state = self.add_dead_nodes_to_edge(state, edge_dead_count)

        # Ensure at least one node is offline
        state = self.ensure_offline_nodes(state)

        return np.rot90(state, np.random.randint(0, 4))

    def ensure_offline_nodes(self, state: State) -> State:
        """Ensures at least one node is 0 (removed)."""
        if np.all(state):  # If no zeros exist
            state[0, 0] = 0  # Set top-left to 0
        return state

    def add_dead_corners(self, state: State) -> State:
        corner_scale = self.scale + 1

        state = self.add_dead_corner(state, corner_scale)

        if random.randint(0, 3) == 0:
            state = self.rotate_n_times(state, 2)  # 180 degree rotation
            state = self.add_dead_corner(state, corner_scale - 2)

        return self.randomize_rotation(state)

    def add_dead_corner(self, state: State, size: int) -> State:
        current_size = size
        for i in range(size):
            if i < current_size:
                if random.randint(0, 1) == 1:
                    current_size -= 1
                for j in range(current_size):
                    if i < state.shape[0] and j < state.shape[1]:
                        state[i, j] = 0
        return state

    def add_center_break(self, state: State) -> State:
        size = state.shape[0]
        max_offset = self.scale
        x_index = random.randint(0, max_offset * 2) - max_offset + size // 2
        x_index = max(0, min(x_index, size - 1))

        max_len = size // 2 - 1
        # Ensure max_len is at least 1 for randint range
        length = random.randint(1, max(1, max_len))

        for i in range(length):
            # Check bounds just in case length is large, though unlikely
            if i < state.shape[1]:
                state[x_index, i] = 0

        return self.randomize_rotation(state)

    def rotate_90_degrees(self, state: State) -> State:
        """Rotates 90 degrees clockwise, matching JS."""
        return np.rot90(state, k=3)

    def randomize_rotation(self, state: State) -> State:
        return self.rotate_n_times(state, random.randint(0, 3))

    def rotate_n_times(self, state: State, rotations: int) -> State:
        temp_state = state
        for _ in range(rotations % 4):
            temp_state = self.rotate_90_degrees(temp_state)
        return temp_state

    def remove_rows(self, state: State) -> State:
        """Corrected to match JS: Remove top rows, then rotate 3 times."""
        rows_to_remove = max(random.randint(-2, self.scale), 1)

        rows_to_remove = min(rows_to_remove, state.shape[0])
        for i in range(rows_to_remove):
            state[i, :] = 0  # Set entire row to 0

        state = self.rotate_n_times(state, 3)

        return state

    def add_dead_nodes_to_edge(self, state: State, max_per_edge: int) -> State:
        size = state.shape[0]
        temp_state = state
        for _ in range(4):
            count = random.randint(0, max_per_edge)
            for _ in range(count):
                y_index = max(random.randint(-2, size - 1), 0)
                y_index = min(y_index, size - 1)
                temp_state[0, y_index] = 0  # Set point on the current 'top' edge to 0
            temp_state = self.rotate_90_degrees(temp_state)  # Rotate for the next edge
        return temp_state

    def print_board(self, state: State) -> None:
        """Debug method to visualize the board state."""
        symbols = {0: "⬛", 1: "⬜"}  # 0 for removed spaces, 1 for valid spaces
        print("Board visualization:")
        header = "  " + " ".join([f"{i:<2}" for i in range(self.board_width)])
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in range(self.board_width):
            row_symbols = [symbols[state[i, j]] for j in range(self.board_width)]
            print(f"{i:<2}|{' '.join(row_symbols)}")


if __name__ == "__main__":
    # Test with different sizes
    for size in [5, 9, 13]:
        print(f"\n--- Generating Board Size: {size}x{size} ---")
        generator = GoStateGenerator(size)
        try:
            board_state = generator.generate_board_state()
            generator.print_board(board_state)
            # Verify ensure_offline_nodes worked if needed
            if np.all(board_state):
                print("ERROR: Board is still all 1s!")
            elif not np.any(board_state == 0):
                print("ERROR: Board contains no 0s!")

        except Exception as e:
            print(f"An error occurred during generation for size {size}: {e}")

    # states = {"top": 0, "left": 0, "right": 0, "bottom": 0}

    # for i in range(10000):
    #     generator = GoStateGenerator(9)
    #     state = generator.generate_board_state()

    #     if np.all(state[0] == 0):
    #         states["top"] += 1
    #     elif np.all(state[8] == 0):
    #         states["bottom"] += 1
    #     elif np.all(state[:, 0] == 0):
    #         states["left"] += 1
    #     elif np.all(state[:, 8] == 0):
    #         states["right"] += 1
    #     else:
    #         pass

    # print(states)

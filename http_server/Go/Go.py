import numpy as np
from collections import deque


def rotate_state(state: list[str]) -> list[str]:
    rotated_state: list[str] = []

    for i in range(len(state)):
        tmp = ""
        for j in range(len(state)):
            tmp += state[j][i]

        rotated_state.append(tmp)
    rotated_state.reverse()
    return rotated_state


def beatify_state(state: list[str], delim="<br>") -> str:
    beautified_state: str = ""
    for i in range(len(state)):
        for j in range(len(state)):
            beautified_state += f"{state[i][j]} "
        beautified_state += delim
    return beautified_state


def rotate_and_beatify(state: list[str], delim: str = "<br>") -> str:
    return beatify_state(rotate_state(state), delim)


class Go:
    def __init__(self, board_width: int, board_height: int, board: list[str]):
        if len(board) != 5:
            raise ValueError("board size must be 5x5")
        self.board_width = board_width
        self.board_height = board_height
        self.board_size = self.board_width * self.board_height
        self.history: list[str] = []

        self.str_board: list[str] = board
        self.board: np.ndarray = self.transform_board(board)

    def decode_action(self, action_idx: int):
        board_size = self.board_width * self.board_height
        if action_idx == board_size:
            return (-1, -1)
        else:
            x = action_idx // self.board_height
            y = action_idx % self.board_height
            return (x, y)

    def encode_action(self, x: int, y: int) -> int:
        return x * self.board_height + y

    def transform_board(self, board: list[str]):
        """
        Converts a list of strings (like [".....", "..X..", ...]) into a numpy array
        with the following encoding:
          '.' -> 0 (empty)
          'X' -> 1 (black)
          'O' -> 2 (white)
          '#' -> 3 (blocked or invalid)
        """
        transformed = np.zeros([5, 5], dtype=int)
        for i, row_str in enumerate(board):
            for j, char in enumerate(row_str):
                if char == ".":
                    transformed[i][j] = 0
                elif char == "X":
                    transformed[i][j] = 1
                elif char == "O":
                    transformed[i][j] = 2
                elif char == "#":
                    transformed[i][j] = 3
        return transformed

    def decode_board(self, board_array: np.ndarray):
        """
        Converts a numpy board array (with 0/1/2/3) back into the
        string-based representation for printing or logging.
        """
        decoded_board = []
        for i in range(board_array.shape[0]):
            tmp = ""
            for j in range(board_array.shape[1]):
                val = board_array[i][j]
                if val == 0:
                    tmp += "."
                elif val == 1:
                    tmp += "X"
                elif val == 2:
                    tmp += "O"
                elif val == 3:
                    tmp += "#"
            decoded_board.append(tmp)
        return decoded_board

    def __str__(self):
        board = self.decode_board(self.board)
        return rotate_and_beatify(
            board, "\n"
        )  # Assuming you have a rotate_and_beatify function

    def state_after_action(self, action: int, is_white: bool) -> tuple[np.ndarray, int]:
        """Returns (new_board_state_as_strings, captures)."""
        if action == self.board_size:  # Pass move
            return self.board.copy(), 0

        x, y = self.decode_action(action)
        new_board = self.board.copy()
        color = 2 if is_white else 1
        captures = 0

        # Place the stone
        new_board[x][y] = color

        # Check for captures (opponent stones)
        visited: set[tuple[int, int]] = set()
        opponent_color = 1 if is_white else 2

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.board_width
                and 0 <= ny < self.board_height
                and new_board[nx][ny] == opponent_color
            ):
                if (nx, ny) not in visited:
                    group = self.find_group(nx, ny, opponent_color, visited, new_board)
                    if not self.has_liberties(group, new_board):
                        # Remove captured group
                        for gx, gy in group:
                            new_board[gx][gy] = 0
                            captures += 1

        return self.decode_board(new_board), captures

    def find_group(
        self, x: int, y: int, color: int, visited: set, board: np.ndarray
    ) -> set[tuple[int, int]]:
        """Find all connected stones of the given color using DFS."""
        if (x, y) in visited:
            return set()
        if not (0 <= x < self.board_width and 0 <= y < self.board_height):
            return set()
        if board[x][y] != color:
            return set()

        visited.add((x, y))
        group = {(x, y)}

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            group.update(self.find_group(x + dx, y + dy, color, visited, board))

        return group

    def has_liberties(self, group: set[tuple[int, int]], board: np.ndarray) -> bool:
        """Check if any position in the group has an empty adjacent point (a liberty)."""
        for x, y in group:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.board_width
                    and 0 <= ny < self.board_height
                    and board[nx][ny] == 0
                ):
                    return True
        return False

    def flood_fill(
        self, x: int, y: int, visited: set
    ) -> tuple[int | None, set[tuple[int, int]]]:
        """
        A helper for territory detection:
          - BFS/DFS from (x, y) to gather all connected empty cells.
          - Track the colors of any stones adjacent to those empty cells.
          - If we find exactly one color, that color "owns" the territory.
          - If we find both black and white, or we run into blocked nodes in a way that
            doesn't result in a single color, it's disputed (None).

        Returns: (color_owner, territory_positions)
                 color_owner = 1 (black), 2 (white), or None (disputed or no single color).
        """

        queue: deque[tuple[int, int]] = deque()
        queue.append((x, y))
        territory = set()
        adjacent_colors = set()

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            # If blocked (3), we skip but do not claim it as territory
            # if self.board[cx][cy] == 3:
            #     continue

            # If its empty -> part of the territory
            if self.board[cx][cy] == 0:
                territory.add((cx, cy))
                # Check neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                        if self.board[nx][ny] == 0:
                            if (nx, ny) not in visited:
                                queue.append((nx, ny))
                        # If it its a stone save its color
                        elif self.board[nx][ny] in [1, 2]:
                            adjacent_colors.add(self.board[nx][ny])

        # If territory touches exactly one color, it belongs to that color
        if len(adjacent_colors) == 1:
            return adjacent_colors.pop(), territory
        # Otherwise its disputed or belongs to no one
        return None, territory

    def get_territory_scores(self) -> tuple[int, int]:
        """
        Finds how many empty intersections belong to black or white via territory.
        Returns (white_territory, black_territory).
        """
        visited: set[tuple[int, int]] = set()
        white_territory = 0
        black_territory = 0

        for x in range(self.board_width):
            for y in range(self.board_height):
                # 0 means node is empty
                if (x, y) not in visited and self.board[x][y] == 0:
                    color_owner, territory = self.flood_fill(x, y, visited)
                    if color_owner == 1:  # black
                        black_territory += len(territory)
                    elif color_owner == 2:  # white
                        white_territory += len(territory)

        return white_territory, black_territory

    def get_score(self, komi: float = 5.5) -> dict:
        """
        Computes the score for white and black, including komi for white.
        - Each stone on the board counts as 1 point
        - Each empty node fully surrounded by one color also counts as territory
        """
        # Count stones directly
        black_pieces = np.count_nonzero(self.board == 1)
        white_pieces = np.count_nonzero(self.board == 2)

        # Get territory counts
        white_territory, black_territory = self.get_territory_scores()

        white_sum = white_pieces + white_territory + komi
        black_sum = black_pieces + black_territory

        return {
            "white": {
                "pieces": white_pieces,
                "territory": white_territory,
                "komi": komi,
                "sum": white_sum,
            },
            "black": {
                "pieces": black_pieces,
                "territory": black_territory,
                "komi": 0,
                "sum": black_sum,
            },
        }


if __name__ == "__main__":
    decoded_board = [".X.OO", "O..XX", ".OX.#", ".OOX.", "O.OX."]  # 9 16.5
    decoded_board = ["..X..", ".....", "#OO.X", ".XO..", "X...."]  # 5, 8.5
    decoded_board = ["..##O", "#XXX.", ".XO.O", ".XXX.", "O..OO"]  # 9 11.5
    decoded_board = ["..#O.", ".XOOO", "XOO.X", ".X..X", "...X."]  # 10, 12.5
    decoded_board = [".XX..", "O.OXX", ".O.O#", ".OOX.", "O.OX."]  # 10 17.5
    go = Go(5, 5, decoded_board)

    print(go.get_score())

import numpy as np
from collections import deque
from copy import deepcopy


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
        self.history: list[np.ndarray] = []

        self.str_board: list[str] = board
        self.state: np.ndarray = self.encode_board(board)
        self.current_player = 1  # black starts

    def __str__(self):
        board = self.decode_board(self.state)
        return rotate_and_beatify(board, "\n")

    def decode_action(self, action_idx: int):
        """
        Converts an action index into board coordinates. The last index is pass action
        Returns:
            tuple: A pair of coordinates (x,y):
                - If action_idx is width*height, returns (-1,-1) representing a pass move
                - Otherwise returns (x,y) coordinates on the board
        """
        board_size = self.board_width * self.board_height
        if action_idx == board_size:
            return (-1, -1)
        else:
            x = action_idx // self.board_height
            y = action_idx % self.board_height
            return (x, y)

    def encode_action(self, x: int, y: int) -> int:
        """
        Converts a board coordinate to an action index
        """
        return x * self.board_height + y

    def encode_board(self, board: list[str]):
        """
        Converts a list of strings (like [".....", "..X..", ...]) into a numpy array
        with the following encoding:
          '.' -> 0 (empty)
          'X' -> 1 (black)
          'O' -> 2 (white)
          '#' -> 3 (disabled)
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
        Converts a numpy board array (with 0/1/2/3) back into the string-based representation.
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

    def state_after_action(self, action: int, is_white: bool) -> np.ndarray:
        """Returns (new_board_state_as_strings, captures)."""
        if action == self.board_size:  # Pass move
            return self.state.copy()

        x, y = self.decode_action(action)
        color = 2 if is_white else 1

        new_state = self.simulate_move(self.state, x, y, color)
        if new_state:
            return new_state
        else:
            return np.array([])

    def flood_fill_territory(
        self, x: int, y: int, visited: set
    ) -> tuple[int | None, set[tuple[int, int]]]:
        """
        A helper for territory detection, territory is how many empty nodes a color surrounds:
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

            # If its empty -> part of the territory
            if self.state[cx][cy] == 0:
                territory.add((cx, cy))
                # Check neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                        if self.state[nx][ny] == 0:
                            if (nx, ny) not in visited:
                                queue.append((nx, ny))
                        # If it its a stone save its color
                        elif self.state[nx][ny] in [1, 2]:
                            adjacent_colors.add(self.state[nx][ny])

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
                if (x, y) not in visited and self.state[x][y] == 0:
                    color_owner, territory = self.flood_fill_territory(x, y, visited)
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
        black_pieces = np.count_nonzero(self.state == 1)
        white_pieces = np.count_nonzero(self.state == 2)

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

    def get_liberties(
        self, state: np.ndarray, x: int, y: int, visited: set[tuple[int, int]]
    ):
        """
        Returns a set of all liberties the color at (x, y) has and the territory
        """
        color: int = state[x][y]
        if color in [0, 3]:
            raise ValueError("Empty Cell or Disabled Cell cannot have liberties")

        queue: deque[tuple[int, int]] = deque()
        queue.append((x, y))

        territory: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            # If its empty -> part of the territory
            if state[cx][cy] == color:
                territory.add((cx, cy))
                # Check neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                        if state[nx][ny] == color:
                            if (nx, ny) not in visited:
                                queue.append((nx, ny))
                        # If its empty (a libery) save it
                        elif state[nx][ny] == 0:
                            liberties.add((nx, ny))

        return liberties, territory

    def simulate_move(
        self, state: np.ndarray, x: int, y: int, color: int
    ) -> np.ndarray | None:
        sim_state = deepcopy(state)
        sim_state[x][y] = color

        # color = 1 => enemy = 2; color = 2 => ememy = 1
        enemy = 3 - color

        # check for capture
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                if sim_state[nx][ny] == enemy:
                    liberties, territory = self.get_liberties(sim_state, nx, ny, set())
                    # capture enemy stones if they have no liberties
                    if len(liberties) == 0:
                        for tx, ty in territory:
                            sim_state[tx][ty] = "."

        # check if placed router has liberties
        libs, _ = self.get_liberties(sim_state, x, y, set())
        if len(libs) == 0:
            # move was actually a suicide
            return None

        # check for repeat
        is_repeat = self.check_state_is_repeat(sim_state)
        if is_repeat:
            return None

        # return some useful data
        return sim_state

    def check_state_is_repeat(self, state: np.ndarray) -> bool:
        prev_len = len(self.history)
        history_set = set([str(h_state) for h_state in self.history])
        history_set.add(str(state))
        if prev_len == len(history_set):
            return True
        return False

    def check_move_is_valid(self, state: np.ndarray, x: int, y: int) -> bool:
        # move is not empty
        if state[x][y] != 0:
            return False

        has_empty_neighbor = False
        has_capturable_enemy = False
        has_safe_friendly = False

        visited: set[tuple[int, int]] = set()
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                # case 1: valid, if cell has empty eighbors
                if state[nx][ny] == 0:
                    has_empty_neighbor = True
                # case 2: any adjacent enemy cell has only one liberty (enemy group will be captured)
                if state[nx][ny] in [1, 2]:
                    libs, _ = self.get_liberties(state, nx, ny, visited)
                    if len(libs) == 1 and state[nx][ny] != self.current_player:
                        has_capturable_enemy = True
                    # or a friendly cell has more than one
                    elif len(libs) > 1 and state[nx][ny] == self.current_player:
                        has_safe_friendly = True

                # suicide if: all adjacent friendly have 1 lib, or all adjacent are enemy

        if has_empty_neighbor or has_capturable_enemy or has_safe_friendly:
            tmp_state = state.copy()
            tmp_state[x][y] = self.current_player
            is_repeat = self.check_state_is_repeat(tmp_state)
            return not is_repeat
        else:
            simulated_board = self.simulate_move(state, x, y, self.current_player)
            if not simulated_board or simulated_board[x][y] != self.current_player:
                return False
            is_repeat = self.check_state_is_repeat(simulated_board)

            return not is_repeat

        # print("no rule applies, must be valid move")
        # return True

    def make_move(self, action: int, is_white: bool) -> np.ndarray:
        state_after_move = self.state_after_action(action, is_white)
        self.state = state_after_move
        self.current_player = 3 - self.current_player
        # return next_state, reward, done
        return self.state

    def get_legal_moves(self, player: int):
        legal_moves: np.ndarray = np.zeros([self.board_width, self.board_height])
        for x in range(self.board_width):
            for y in range(self.board_height):
                is_legal = self.check_move_is_valid(self.state, x, y)
                legal_moves[x][y] = is_legal
        return legal_moves


if __name__ == "__main__":
    decoded_board = [".X...", ".X.XO", "#XO..", ".XOOO", ".XO.#"]
    decoded_board = [".XO..", "XXOO.", "OOO.#", "..OXX", ".X.XX"]
    decoded_board = [".OXO.", ".O.O#", "#.O..", "....X", ".XXX#"]
    decoded_board = [".OX..", ".O.O.", "..O..", ".....", ".XXX."]
    decoded_board = [".OX.O", ".OXO.", "..O..", ".....", ".XXX."]
    decoded_board = [
        ".OX.O",
        ".OXOX",
        "..O..",
        ".....",
        ".XXXO",
    ]  # place at 0, 3; legal for both
    decoded_board = [
        "#XO.X",
        "#XOXX",
        "#.XOO",
        "#OO.O",
        "#...X",
    ]  # 0,3 -> both legal, 2,1 -> only for white
    decoded_board = ["#....", "#...#", "#....", "#.O.X", "#.#X."]
    go = Go(5, 5, decoded_board)
    go.current_player = 2

    # vis = set()
    # print(go.get_liberties(go.state, 3, 2, vis))
    print(go)
    print(go.check_move_is_valid(go.state, 4, 4))


# done: score
# check move is valid

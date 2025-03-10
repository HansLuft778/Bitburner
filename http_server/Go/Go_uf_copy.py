import time # pyright: ignore
from collections import deque

import numpy as np
from typing import Any, cast

State = np.ndarray[Any, np.dtype[np.int8]]


class UnionFind:
    def __init__(
        self,
        parent: np.ndarray[Any, np.dtype[np.int8]],
        colors: np.ndarray[Any, np.dtype[np.int8]],
        rank: np.ndarray[Any, np.dtype[np.int8]],
        stones: list[set[int]],
        liberties: list[set[int]],
        board_size: int,
    ):
        self.parent = parent
        self.colors = colors
        self.rank = rank
        self.stones = stones
        self.liberties = liberties
        self.board_height = board_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnionFind):
            return NotImplemented
        return (
            (self.parent == other.parent).all()
            and (self.colors == other.colors).all()
            and (self.rank == other.rank).all()
            and self.stones == other.stones
            and self.liberties == other.liberties
        )

    def __str__(self):
        parent_str = ", ".join(map(str, self.parent))
        colors_str = ", ".join(map(str, self.colors))
        rank_str = ", ".join(map(str, self.rank))
        return f"parent: [{parent_str}],\ncolors: [{colors_str}],\nrank: [{rank_str}],\nstones: {self.stones},\nliberties: {self.liberties}"

    def find(self, i: int) -> int:
        """
        Find with path compression.
        Note: Path compression changes the tree structure but doesn't affect ranks
        """
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    # Remove or comment out the old union() method

    # Rename union_2 to union and use it as the primary implementation
    def union(self, a: int, b: int, state: State) -> None:
        """
        Unites two groups of stones together using union by rank.
        Rank represents the upper bound of the height of the tree.
        """
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        assert (
            self.colors[root_a] == self.colors[root_b]
        ), f"Colors must be the same: {root_a} {root_b}"
        assert root_a != root_b, f"Roots must be different: {root_a} {root_b}"
        assert root_a != -1 and root_b != -1, f"Roots must be valid: {root_a} {root_b}"

        # Union by rank - attach smaller rank tree under root of higher rank tree
        if self.rank[root_a] > self.rank[root_b]:
            # No rank change needed when attaching smaller to larger
            self.parent[root_b] = root_a
            self.stones[root_a].update(self.stones[root_b])
            self.stones[root_b].clear()
            self.liberties[root_a].update(self.liberties[root_b])
            self.liberties[root_b].clear()
            self.rank[root_b] = -1
        else:
            self.parent[root_a] = root_b
            self.stones[root_b].update(self.stones[root_a])
            self.stones[root_a].clear()
            self.liberties[root_b].update(self.liberties[root_a])
            self.liberties[root_a].clear()
            # Increment rank of root_b only if ranks were equal
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_b] += 1
            self.rank[root_a] = -1

        new_root = self.find(a)
        self.liberties[new_root].clear()

        for stone in self.stones[new_root]:
            sx, sy = stone // self.board_height, stone % self.board_height
            # Check all four adjacent positions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < self.board_height and 0 <= ny < self.board_height:
                    if state[nx][ny] == 0:  # Empty = liberty
                        liberty_pos = nx * self.board_height + ny
                        self.liberties[new_root].add(liberty_pos)

    @staticmethod
    def get_uf_from_state(state: State) -> "UnionFind":

        width: int = state.shape[0]

        parent = np.full(width * width, -1, dtype=np.int8)
        colors = np.full(width * width, -1, dtype=np.int8)
        rank = np.full(width * width, -1, dtype=np.int8)
        stones: list[set[int]] = [set() for _ in range(width * width)]
        liberties: list[set[int]] = [set() for _ in range(width * width)]
        uf = UnionFind(parent, colors, rank, stones, liberties, width)

        for x in range(width):
            for y in range(width):
                idx = x * width + y
                if state[x][y] == 3:
                    continue
                if state[x][y] == 0:
                    # check if the empty cell is a liberty
                    for dx, dy in [(0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < width:
                            nidx = nx * width + ny
                            if state[nx][ny] in (1, 2):
                                n_root = uf.find(nidx)
                                liberties[n_root].add(idx)
                    continue

                parent[idx] = idx
                colors[idx] = state[x][y]
                stones[idx] = set([idx])
                liberties[idx] = set()
                rank[idx] = 0

                color = state[x][y]
                enemy = 3 - color
                idx_root = idx
                for dx, dy in [(0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < width:
                        nidx = nx * width + ny
                        # neighbor is empty
                        if state[nx][ny] == 0:
                            liberties[idx_root].add(nidx)
                        # neighbor is same color
                        elif colors[nidx] == color:
                            uf.union(idx, nidx, state)
                            idx_root = uf.find(idx)
                        # neighbor is enemy
                        elif uf.colors[nidx] == enemy:
                            enemy_group = uf.find(nidx)
                            uf.liberties[enemy_group].discard(idx)

        return uf

    def copy(self):
        return UnionFind(
            self.parent.copy(),
            self.colors.copy(),
            self.rank.copy(),
            [s.copy() for s in self.stones],
            [s.copy() for s in self.liberties],
            self.board_height,
        )

    def print(self):
        print("parent: ", ", ".join(map(str, self.parent)))
        print("colors: ", ", ".join(map(str, self.colors)))
        print("rank: ", ", ".join(map(str, self.rank)))
        print("stones: ", [s if s else cast(set[int], set()) for s in self.stones])
        print(
            "liberties: ", [l if l else cast(set[int], set()) for l in self.liberties]
        )


def rotate_state(state: list[str]) -> list[str]:
    rotated_state: list[str] = []

    for i in range(len(state)):
        tmp = ""
        for j in range(len(state)):
            tmp += state[j][i]

        rotated_state.append(tmp)
    rotated_state.reverse()
    return rotated_state


def beatify_state(state: list[str], delim: str = "<br>") -> str:
    beautified_state: str = ""
    for i in range(len(state)):
        for j in range(len(state)):
            beautified_state += f"{state[i][j]} "
        beautified_state += delim
    return beautified_state


def rotate_and_beatify(state: list[str], delim: str = "<br>") -> str:
    return beatify_state(rotate_state(state), delim)


class Go_uf:
    def __init__(self, board_width: int, state: State, komi: float):

        self.board_width = board_width
        self.board_height = board_width
        self.board_size = self.board_width * self.board_height
        self.history: list[State] = []
        self.previous_action = -1

        self.state: State = state
        self.current_player = 1  # black starts
        self.komi = komi

        # union find data structure
        parent = np.full(self.board_width * self.board_height, -1, dtype=np.int8)
        colors = np.full(self.board_width * self.board_height, -1, dtype=np.int8)
        rank = np.full(self.board_width * self.board_height, -1, dtype=np.int8)
        stones: list[set[int]] = [
            set() for _ in range(5 * 5)
        ]  # set of which stones are in the same group of a root
        liberties: list[set[int]] = [
            set() for _ in range(self.board_width * self.board_width)
        ]
        self.uf = UnionFind(parent, colors, rank, stones, liberties, self.board_width)

    def __str__(self):
        board = self.decode_state(self.state)
        return rotate_and_beatify(board, "\n")

    def decode_action(self, action_idx: int):
        """
        Converts an action index into board coordinates. The last index is pass action
        Returns:
            tuple: A pair of coordinates (x,y):
                - If action_idx is width*height, returns (-1,-1) representing a pass move
                - Otherwise returns (x,y) coordinates on the board
        """
        if action_idx == self.board_width * self.board_height:
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

    def encode_state(self, state: list[str]):
        """
        Converts a list of strings (like [".....", "..X..", ...]) into a numpy array
        with the following encoding:
          '.' -> 0 (empty)
          'X' -> 1 (black)
          'O' -> 2 (white)
          '#' -> 3 (disabled)
        """
        transformed = np.zeros([5, 5], dtype=np.int8)
        for i, row_str in enumerate(state):
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

    def decode_state(self, state: State) -> list[str]:
        """
        Converts a numpy board array (with 0/1/2/3) back into the string-based representation.
        """
        decoded_board: list[str] = []
        for i in range(state.shape[0]):
            tmp = ""
            for j in range(state.shape[1]):
                val = state[i][j]
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

    def state_after_action(
        self,
        action: int,
        is_white: bool,
        provided_state: State,
        uf: UnionFind,
        additional_history: list[State] = [],
    ) -> tuple[State, UnionFind]:
        if action == self.board_size:  # Pass move
            return provided_state, uf

        color = 2 if is_white else 1

        new_state, new_uf = self.simulate_move(
            provided_state,
            uf,
            action,
            color,
            additional_history,
        )
        x, y = self.decode_action(action)
        new_state_original = self.simulate_move_original(
            provided_state, x, y, color, additional_history
        )

        if new_state is None:
            assert (
                new_state_original is None
            ), f"States must be equal: {new_state_original}"
        if new_state_original is None:
            assert new_state is None, f"States must be equal: {new_state}"
        if new_state is not None and new_state_original is not None:
            assert np.array_equal(
                new_state, new_state_original
            ), f"States must be equal: {new_state} {new_state_original}"
        if new_state_original is not None:
            return new_state_original, new_uf
        else:
            return np.array([]), uf

    def flood_fill_territory(
        self, x: int, y: int, visited: set[tuple[int, int]]
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
        territory: set[tuple[int, int]] = set()
        adjacent_colors: set[int] = set()

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

        # if territory is too large, its not valid
        if len(territory) > 2 * self.board_width:
            return None, territory

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

    def get_score(self, komi: float) -> dict[str, dict[str, float]]:
        """
        Computes the score for white and black, including komi for white.
        - Each stone on the board counts as 1 point
        - Each empty node fully surrounded by one color also counts as territory
        """
        # Count stones directly
        black_pieces = np.sum(self.state == 1)
        white_pieces = np.sum(self.state == 2)

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
        self, state: State, x: int, y: int, visited: set[tuple[int, int]]
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

    def simulate_move_original(
        self,
        state: State,
        x: int,
        y: int,
        color: int,
        additional_history: list[State] = [],
    ) -> State | None:
        if state[x][y] != 0:
            return None

        sim_state = state.copy()
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
                            sim_state[tx][ty] = 0

        # check if placed router has liberties
        libs, _ = self.get_liberties(sim_state, x, y, set())
        if len(libs) == 0:
            # move was actually a suicide
            return None

        # check for repeat
        is_repeat = self.check_state_is_repeat(sim_state, additional_history)
        if is_repeat:
            return None

        return sim_state

    def simulate_move(
        self,
        state: State,
        uf: UnionFind,
        action: int,
        color: int,
        # i_know_its_legal: bool,
        additional_history: list[State] = [],
    ) -> tuple[State | None, UnionFind]:
        x, y = self.decode_action(action)
        if state[x][y] != 0:
            return None, uf

        sim_state = state.copy()
        uf_before = uf.copy()

        sim_state[x][y] = color

        # color = 1 => enemy = 2; color = 2 => ememy = 1
        enemy = 3 - color

        # Initialize the new stone's data
        uf_before.parent[action] = action  # stone is its own root initially
        uf_before.colors[action] = color
        uf_before.stones[action] = set([action])
        uf_before.liberties[action] = set()
        uf_before.rank[action] = 0  # new single nodes start with rank 0

        action_root = action
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                nidx = self.encode_action(nx, ny)
                # neighbor is empty
                if state[nx][ny] == 0:
                    uf_before.liberties[action_root].add(nidx)
                # neighbor is same color
                elif uf_before.colors[nidx] == color:
                    uf_before.union(action, nidx, sim_state)
                    action_root = uf_before.find(action)
                # neighbor is enemy
                elif uf_before.colors[nidx] == enemy:
                    enemy_group = uf_before.find(nidx)

                    uf_before.liberties[enemy_group].discard(action)

                    # capture enemy group if no liberties
                    if len(uf_before.liberties[enemy_group]) == 0:
                        enemy_group_stones = list(uf_before.stones[enemy_group])
                        for stone in enemy_group_stones:
                            sx, sy = self.decode_action(stone)
                            sim_state[sx][sy] = 0
                            # Record the captured stone's state
                            uf_before.colors[stone] = -1
                            uf_before.parent[stone] = -1
                            # uf_before.stones[stone].clear()
                            # uf_before.liberties[
                            #     stone
                            # ].clear()  # potentially redundant (add assert to check)
                            uf_before.rank[stone] = -1
                        uf_before.stones[enemy_group].clear()
                        uf_before.liberties[enemy_group].clear()

                        # update liberties of neighboring groups
                        for stone in enemy_group_stones:
                            sx, sy = self.decode_action(stone)
                            for ddx, ddy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                snx, sny = sx + ddx, sy + ddy
                                if (
                                    0 <= snx < self.board_width
                                    and 0 <= sny < self.board_height
                                ):
                                    snidx = self.encode_action(snx, sny)
                                    if uf_before.colors[snidx] == color:
                                        neighbor_root = uf_before.find(snidx)
                                        uf_before.liberties[neighbor_root].add(stone)

        is_consitent = self.verify_uf_consistency(sim_state, uf_before)
        assert is_consitent, f"UF is not consistent with the board state"

        # check if placed stone has liberties
        if len(uf_before.liberties[action_root]) == 0:
            # move was actually a suicide
            return None, uf_before

        # check for repeat
        is_repeat = self.check_state_is_repeat(sim_state, additional_history)
        if is_repeat:
            # move was actually a repeat
            return None, uf_before

        return sim_state, uf_before

    def verify_uf_consistency(self, state: State, uf: UnionFind) -> bool:
        """Verify that the UnionFind structure is consistent with the board state"""
        for x in range(self.board_width):
            for y in range(self.board_height):
                idx = self.encode_action(x, y)
                if state[x][y] in [1, 2]:  # Stone exists
                    if uf.colors[idx] != state[x][y]:
                        return False
                else:  # Empty or disabled
                    if uf.colors[idx] != -1:
                        return False
        return True

    def check_state_is_repeat(
        self, state: State, additional_history: list[State] = []
    ) -> bool:
        state_bytes = state.tobytes()
        history_bytes = [e.tobytes() for e in self.history]
        additional_bytes = [e.tobytes() for e in additional_history]

        return state_bytes in history_bytes or state_bytes in additional_bytes

    def make_move(self, action: int, is_white: bool) -> tuple[State, int, bool]:
        state_after_move, uf_after_move = self.state_after_action(
            action, is_white, self.state, self.uf
        )
        game_ended = self.has_game_ended(action, is_white, self.state, self.uf)

        self.state = state_after_move
        self.uf = uf_after_move
        self.current_player = 3 - self.current_player

        # update history and prev action
        self.history.append(self.state.copy())
        self.previous_action = action

        # outcome is 1 black won, -1 white won, 0 not ended
        outcome = 0
        if game_ended:
            score = self.get_score(self.komi)
            print(f"score: {score}")
            outcome = 1 if score["black"]["sum"] > score["white"]["sum"] else -1

        return self.state, outcome, game_ended

    def get_valid_moves(
        self,
        state: State,
        uf: UnionFind,
        is_white: bool,
        history: list[State] = [],
    ) -> np.ndarray[Any, np.dtype[np.bool_]]:
        player = 2 if is_white else 1

        legal_moves: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros_like(
            state, dtype=bool
        )
        empty_mask = state == 0
        empty_positions = np.where(empty_mask)
        for x, y in zip(empty_positions[0], empty_positions[1]):
            action = self.encode_action(int(x), int(y))
            new = self.simulate_move(
                state,
                uf,
                action,
                player,
                history,
            )[0]
            old = self.simulate_move_original(state, int(x), int(y), player, history)
            if new is not None and old is not None:
                assert np.array_equal(new, old), f"States must be equal"
            else:
                assert new is None and old is None, f"States must be equal"

            if old is not None:
                legal_moves[x][y] = True
        return legal_moves

    def has_game_ended(
        self,
        action: int,
        is_white: bool,
        state: State,
        uf: UnionFind,
        additional_history: list[State] = [],
    ) -> bool:
        # double pass
        if self.previous_action == action == self.board_width * self.board_height:
            return True

        # previous pass, current has no valid moves
        valid = self.get_valid_moves(state, uf, is_white, additional_history)
        if (
            self.previous_action == self.board_width * self.board_height
            and np.sum(valid) == 0
        ):
            return True

        # board is full
        has_empty_node = np.any(state == 0)
        if has_empty_node:
            return False

        return False

    def get_history(self) -> list[State]:
        return self.history


if __name__ == "__main__":
    decoded_board = np.array(
        [
            [3, 0, 1, 2, 2],
            [0, 1, 1, 2, 0],
            [0, 2, 0, 1, 2],
            [1, 0, 0, 2, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    uf = UnionFind.get_uf_from_state(decoded_board)
    print(uf)

    # go = Go_uf(5, decoded_board, 5.5)

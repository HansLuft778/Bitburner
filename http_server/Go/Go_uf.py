import time
from collections import deque

import numpy as np


class UnionFind:
    def __init__(
        self,
        parent: np.ndarray,
        colors: np.ndarray,
        rank: np.ndarray,
        stones: list[set[int]],
        liberties: list[set[int]],
    ):
        self.parent = parent
        self.colors = colors
        self.rank = rank
        self.stones = stones
        self.liberties = liberties
        self.board_height = 5

    def __eq__(self, value) -> bool:
        return (
            (self.parent == value.parent).all()
            and (self.colors == value.colors).all()
            and (self.rank == value.rank).all()
            and self.stones == value.stones
            and self.liberties == value.liberties
        )

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
    def union(self, a: int, b: int, state: np.ndarray, undo_stack: list[tuple]) -> None:
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

        undo_stack.append(
            (
                "union",
                root_a,
                root_b,
                self.parent[root_a],
                self.parent[root_b],
                self.rank[root_a],
                self.rank[root_b],
                self.stones[root_a].copy(),
                self.stones[root_b].copy(),
                self.liberties[root_a].copy(),
                self.liberties[root_b].copy(),
            )
        )

        # Union by rank - attach smaller rank tree under root of higher rank tree
        if self.rank[root_a] > self.rank[root_b]:
            # No rank change needed when attaching smaller to larger
            self.parent[root_b] = root_a
            self.stones[root_a].update(self.stones[root_b])
            self.stones[root_b].clear()
            self.liberties[root_a].update(self.liberties[root_b])
            self.liberties[root_b].clear()
            self.colors[root_a] = self.colors[root_b]  # probably not needed
        else:
            self.parent[root_a] = root_b
            self.stones[root_b].update(self.stones[root_a])
            self.stones[root_a].clear()
            self.liberties[root_b].update(self.liberties[root_a])
            self.liberties[root_a].clear()
            self.colors[root_b] = self.colors[root_a]  # probably not needed
            # Increment rank of root_b only if ranks were equal
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_b] += 1

        new_root = self.find(a)
        self.liberties[new_root].clear()
        # for lib_idx in list(self.liberties[new_root]):
        #     lx =
        #     ly = lib_idx % self.board_height
        #     if state[lx][ly] != 0:
        #         undo_stack.append(("remove_liberty", new_root, lib_idx))
        #         self.liberties[new_root].remove(lib_idx)
        for stone in self.stones[new_root]:
            sx, sy = stone // self.board_height, stone % self.board_height
            # Check all four adjacent positions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < self.board_height and 0 <= ny < self.board_height:
                    if state[nx][ny] == 0:  # Empty = liberty
                        liberty_pos = nx * self.board_height + ny
                        self.liberties[new_root].add(liberty_pos)

    def undo_changes(self, undo_stack: list) -> None:
        """
        Revert all changes recorded in the undo stack.
        Process in reverse order (LIFO).
        """
        while undo_stack:
            action = undo_stack.pop()

            if action[0] == "union":
                (
                    _,
                    root_a,
                    root_b,
                    parent_a,
                    parent_b,
                    rank_a,
                    rank_b,
                    stones_a,
                    stones_b,
                    liberties_a,
                    liberties_b,
                ) = action

                # Restore parents
                self.parent[root_a] = parent_a
                self.parent[root_b] = parent_b

                # Restore rank
                self.rank[root_a] = rank_a
                self.rank[root_b] = rank_b

                # Restore stones and liberties
                self.stones[root_a] = stones_a
                self.stones[root_b] = stones_b
                self.liberties[root_a] = liberties_a
                self.liberties[root_b] = liberties_b

            elif action[0] == "remove_liberty":
                _, root, lib_idx = action
                self.liberties[root].add(lib_idx)

            elif action[0] == "initialize_stone":
                _, action_idx = action
                self.parent[action_idx] = -1
                self.colors[action_idx] = -1
                self.stones[action_idx].clear()
                self.liberties[action_idx].clear()
                self.rank[action_idx] = -1

            elif action[0] == "update_liberty":
                _, group_idx, liberty_idx, was_present = action
                if was_present:
                    self.liberties[group_idx].add(liberty_idx)
                else:
                    self.liberties[group_idx].discard(liberty_idx)

            elif action[0] == "capture_stone":
                _, stone_idx, color, parent, rank = action
                x, y = stone_idx // self.board_height, stone_idx % self.board_height
                self.colors[stone_idx] = color
                self.parent[stone_idx] = parent
                self.rank[stone_idx] = rank

    def copy(self):
        return UnionFind(
            self.parent.copy(),
            self.colors.copy(),
            self.rank.copy(),
            [s.copy() for s in self.stones],
            [s.copy() for s in self.liberties],
        )

    def print(self):
        print("parent: ", self.parent)
        print("colors: ", self.colors)
        print("rank: ", self.rank)
        print("stones: ", [s if s else set() for s in self.stones])
        print("liberties: ", [l if l else set() for l in self.liberties])


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


class Go_uf:
    def __init__(self, board_width: int, state: np.ndarray, komi: float):

        self.board_width = board_width
        self.board_height = board_width
        self.board_size = self.board_width * self.board_height
        self.history: list[np.ndarray] = []
        self.previous_action = -1

        self.state: np.ndarray = state
        self.current_player = 1  # black starts

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
        self.uf = UnionFind(parent, colors, rank, stones, liberties)

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

    def decode_state(self, state: np.ndarray) -> list[str]:
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
        provided_state: np.ndarray,
        uf: UnionFind,
        additional_history: list[np.ndarray] = [],
    ) -> tuple[np.ndarray, UnionFind]:
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

        assert (
            new_state is not None
            and new_state_original is not None
            and np.array_equal(new_state, new_state_original)
        ), f"States must be equal"
        if new_state_original is not None:
            return new_state_original, new_uf
        else:
            return np.array([]), uf

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

    def get_score(self, komi: float = 5.5) -> dict:
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

    def simulate_move_original(
        self,
        state: np.ndarray,
        x: int,
        y: int,
        color: int,
        additional_history: list[np.ndarray] = [],
    ) -> np.ndarray | None:
        if state[x][y] != 0:
            return None

        sim_state = state.copy()
        sim_state[x][y] = color

        # color = 1 => enemy = 2; color = 2 => ememy = 1
        enemy = 3 - color

        start_time = time.time()

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

        end_time = time.time()

        return sim_state  # , end_time - start_time

    def simulate_move(
        self,
        state: np.ndarray,
        uf: UnionFind,
        action: int,
        color: int,
        # i_know_its_legal: bool,
        additional_history: list[np.ndarray] = [],
    ) -> tuple[np.ndarray | None, UnionFind]:
        x, y = self.decode_action(action)
        if state[x][y] != 0:
            return None, uf

        sim_state = state.copy()
        uf_before = uf.copy()
        undo_stack: list[tuple] = []

        sim_state[x][y] = color

        # color = 1 => enemy = 2; color = 2 => ememy = 1
        enemy = 3 - color

        # Initialize the new stone's data
        undo_stack.append(("initialize_stone", action))
        uf_before.parent[action] = action  # stone is its own root initially
        uf_before.colors[action] = color
        uf_before.stones[action] = set([action])
        uf_before.liberties[action] = set()
        uf_before.rank[action] = 0  # new single nodes start with rank 0

        start_time = time.time()

        action_root = action
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            nidx = self.encode_action(nx, ny)
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                # neighbor is empty
                if state[nx][ny] == 0:
                    undo_stack.append(("update_liberty", action_root, nidx, False))
                    uf_before.liberties[action_root].add(nidx)
                # neighbor is same color
                elif uf_before.colors[nidx] == color:
                    uf_before.union(action, nidx, sim_state, undo_stack)
                    action_root = uf_before.find(action)
                # neighbor is enemy
                elif uf_before.colors[nidx] == enemy:
                    enemy_group = uf_before.find(nidx)
                    was_present = nidx in uf_before.liberties[enemy_group]
                    undo_stack.append(
                        ("update_liberty", enemy_group, action, was_present)
                    )
                    uf_before.liberties[enemy_group].discard(action)

                    # capture enemy group if no liberties
                    if len(uf_before.liberties[enemy_group]) == 0:
                        enemy_group_stones = list(uf_before.stones[enemy_group])
                        for stone in enemy_group_stones:
                            sx, sy = self.decode_action(stone)
                            sim_state[sx][sy] = 0
                            # Record the captured stone's state
                            undo_stack.append(
                                (
                                    "capture_stone",
                                    stone,
                                    uf_before.colors[stone],
                                    uf_before.parent[stone],
                                    uf_before.rank[stone],
                                )
                            )
                            uf_before.colors[stone] = -1
                            uf_before.parent[stone] = -1
                            uf_before.stones[stone].clear()
                            uf_before.liberties[
                                stone
                            ].clear()  # potentially redundant (add assert to check)
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
                                    if uf_before.colors[snidx] in [1, 2]:
                                        neighbor_root = uf_before.find(snidx)
                                        was_present = (
                                            stone in uf_before.liberties[neighbor_root]
                                        )
                                        undo_stack.append(
                                            (
                                                "update_liberty",
                                                neighbor_root,
                                                stone,
                                                was_present,
                                            )
                                        )
                                        uf_before.liberties[neighbor_root].add(stone)

        # check if placed stone has liberties
        if len(uf_before.liberties[action_root]) == 0:
            # move was actually a suicide
            # print("su")
            # uf_before.undo_changes(undo_stack)
            # uf = uf_before
            # assert uf == uf_before
            return None, uf_before

        # check for repeat
        is_repeat = self.check_state_is_repeat(sim_state, additional_history)
        if is_repeat:
            # move was actually a repeat
            # print("re")
            # uf_before.undo_changes(undo_stack)
            # uf = uf = uf_before
            # assert uf == uf_before
            return None, uf_before

        end_time = time.time()

        return sim_state, uf_before  # , end_time - start_time

    def check_state_is_repeat(
        self, state: np.ndarray, additional_history: list[np.ndarray] = []
    ) -> bool:
        state_bytes = state.tobytes()
        history_bytes = [e.tobytes() for e in self.history]
        additional_bytes = [e.tobytes() for e in additional_history]

        return state_bytes in history_bytes or state_bytes in additional_bytes

    def make_move(self, action: int, is_white: bool) -> tuple[np.ndarray, float, bool]:
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
            score = self.get_score()
            print(f"score: {score}")
            outcome = 1 if score["black"]["sum"] > score["white"]["sum"] else -1

        return self.state, outcome, game_ended

    def get_valid_moves(
        self,
        state: np.ndarray,
        uf: UnionFind,
        is_white: bool,
        history=[],
    ) -> np.ndarray:
        player = 2 if is_white else 1

        legal_moves: np.ndarray = np.zeros_like(state, dtype=bool)
        empty_mask = state == 0
        empty_positions = np.where(empty_mask)
        for x, y in zip(empty_positions[0], empty_positions[1]):
            action = self.encode_action(x, y)
            new = self.simulate_move(
                state,
                uf,
                action,
                player,
                history,
            )[0]
            old = self.simulate_move_original(state, x, y, player, history)
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
        state: np.ndarray,
        uf: UnionFind,
        additional_history: list[np.ndarray] = [],
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

    def get_history(self) -> list[np.ndarray]:
        return self.history


if __name__ == "__main__":
    decoded_board = np.array(
        [
            [3, 0, 2, 0, 0],
            [3, 1, 2, 0, 3],
            [3, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [3, 1, 2, 0, 3],
        ]
    )

    uf = UnionFind(
        np.array(
            [
                -1,
                -1,
                2,
                -1,
                -1,
                -1,
                6,
                2,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                21,
                22,
                23,
                -1,
            ]
        ),
        np.array(
            [
                -1,
                -1,
                2,
                -1,
                -1,
                -1,
                1,
                2,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                2,
                1,
                -1,
            ]
        ),
        np.array(
            [
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                0,
                0,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                0,
                0,
                -1,
            ]
        ),
        [
            set(),
            set(),
            {2, 7},
            set(),
            set(),
            set(),
            {6},
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            {21},
            {22},
            {23},
            set(),
        ],
        [
            set(),
            set(),
            {1, 3, 8, 12},
            set(),
            set(),
            set(),
            {1, 11},
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            {16},
            {17},
            {18},
            set(),
        ],
    )

    go = Go_uf(5, decoded_board, 5.5)
    go.state = decoded_board
    go.uf = uf

    r = go.simulate_move(decoded_board, uf, 17, 1)
    print(r)

    x, y = go.decode_action(17)
    r1 = go.simulate_move_original(decoded_board, x, y, 1)
    print(r1)

    # timings = []
    # for i in range(1000000):
    #     start = time.time()
    #     b, timeing = go.simulate_move_original(decoded_board, x, y, 1)
    #     end = time.time()
    #     timings.append((end - start, timeing))

    # avg_total = sum([t[0] for t in timings]) / len(timings)
    # avg_timeing = sum([t[1] for t in timings]) / len(timings)
    # sum_total = sum([t[0] for t in timings])
    # sum_timeing = sum([t[1] for t in timings])
    # print(avg_total, avg_timeing)
    # print(sum_total, sum_timeing)

    # timings = []
    # for i in range(1000000):
    #     start = time.time()
    #     b, uf, timeing = go.simulate_move(decoded_board, uf, 17, 1)
    #     end = time.time()
    #     timings.append((end - start, timeing))

    # avg_total = sum([t[0] for t in timings]) / len(timings)
    # avg_timeing = sum([t[1] for t in timings]) / len(timings)
    # sum_total = sum([t[0] for t in timings])
    # sum_timeing = sum([t[1] for t in timings])
    # print(avg_total, avg_timeing)
    # print(sum_total, sum_timeing)

    # print(r)  # 5.291655799985165

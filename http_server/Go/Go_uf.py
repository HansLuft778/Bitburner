import time  # pyright: ignore
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any
import numba  # type: ignore

import numpy as np

from Go.Zobrist_hash import ZobristHash

State = np.ndarray[Any, np.dtype[np.int8]]


class UndoActionType(Enum):
    SET_STATE = 1
    SET_PARENT = 2
    SET_RANK = 3
    SET_STONES = 4
    SET_LIBERTIES = 5
    SET_COLOR = 6


@dataclass
class UndoAction:
    action_type: UndoActionType
    position: int  # Can be board position or group index
    value: Any  # The value before change


@numba.njit()  # type: ignore
def get_bit_indices(mask: np.uint64) -> np.ndarray[Any, np.dtype[np.int32]]:
    indices = np.empty(64, np.int32)
    count = 0
    for i in range(64):
        if (mask >> i) & 1:
            indices[count] = i
            count += 1
    return indices[:count]


@numba.njit()  # type: ignore
def calculate_group_liberties(stones_mask: np.int64, state: State, board_size: int) -> np.int64:
    """Calculate liberties for a group of stones with fast bit manipulation"""
    liberties_mask = np.int64(0)

    # Process each stone in the group
    for pos in range(64):  # Assuming 64-bit integer
        if not (stones_mask & (np.int64(1) << pos)):
            continue

        x = pos // board_size
        y = pos % board_size

        # Check each neighbor
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # If neighbor is within bounds and empty, it's a liberty
            if 0 <= nx < board_size and 0 <= ny < board_size and state[nx, ny] == 0:
                lib_pos = nx * board_size + ny
                liberties_mask |= np.int64(1) << lib_pos

    return liberties_mask


def add_stone_to_group(array: np.ndarray[Any, np.dtype[np.int64]], group_idx: int, pos: int):
    array[group_idx] |= 1 << pos


def remove_stone_from_group(array: np.ndarray[Any, np.dtype[np.int64]], group_idx: int, pos: int):
    array[group_idx] &= ~(1 << pos)


class UnionFind:
    def __init__(
        self,
        state: State,
        parent: np.ndarray[Any, np.dtype[np.int8]],
        colors: np.ndarray[Any, np.dtype[np.int8]],
        rank: np.ndarray[Any, np.dtype[np.int8]],
        stones: np.ndarray[Any, np.dtype[np.int64]],  # list[set[int]],
        liberties: np.ndarray[Any, np.dtype[np.int64]],  # list[set[int]],
        board_size: int,
    ):
        self.state = state
        self.hash = np.uint64(0)

        self.parent = parent
        self.colors = colors
        self.rank = rank
        self.stones = stones
        self.liberties = liberties
        self.board_size = board_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnionFind):
            return NotImplemented
        return (
            np.array_equal(self.parent, other.parent)
            and np.array_equal(self.colors, other.colors)
            and np.array_equal(self.rank, other.rank)
            and np.array_equal(self.stones, other.stones)
            and np.array_equal(self.liberties, other.liberties)
        )

    def __str__(self):
        parent_str = ", ".join(map(str, self.parent))
        colors_str = ", ".join(map(str, self.colors))
        rank_str = ", ".join(map(str, self.rank))
        stone_str = ", ".join(map(str, self.stones))
        liberties_str = ", ".join(map(str, self.liberties))
        return f"parent: [{parent_str}],\ncolors: [{colors_str}],\nrank: [{rank_str}],\nstones: {stone_str},\nliberties: {liberties_str}"

    def find(self, i: int) -> int:
        """
        Find with path compression.
        Note: Path compression changes the tree structure but doesn't affect ranks
        """
        if i < 0 or i >= len(self.parent):
            return -1
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def find_no_compression(self, i: int) -> int:
        """Find without path compression, for simulations"""
        if i < 0 or i >= self.board_size * self.board_size:
            return -1
        if self.parent[i] == i:
            return i
        return self.find_no_compression(self.parent[i])

    # Remove or comment out the old union() method

    # Rename union_2 to union and use it as the primary implementation
    def union(self, a: int, b: int) -> None:
        """
        Unites two groups of stones together using union by rank.
        Rank represents the upper bound of the height of the tree.
        """
        root_a = self.find_no_compression(a)
        root_b = self.find_no_compression(b)
        if root_a == root_b:
            return

        assert self.colors[root_a] == self.colors[root_b], f"Colors must be the same: {root_a} {root_b}"
        assert root_a != root_b, f"Roots must be different: {root_a} {root_b}"
        assert root_a != -1 and root_b != -1, f"Roots must be valid: {root_a} {root_b}"

        # Union by rank - attach smaller rank tree under root of higher rank tree
        if self.rank[root_a] > self.rank[root_b]:
            # No rank change needed when attaching smaller to larger
            self.parent[root_b] = root_a
            self.stones[root_a] |= self.stones[root_b]
            self.stones[root_b] = 0
            self.liberties[root_a] |= self.liberties[root_b]
            self.liberties[root_b] = 0
            self.rank[root_b] = -1
        else:
            self.parent[root_a] = root_b
            self.stones[root_b] |= self.stones[root_a]
            self.stones[root_a] = 0
            self.liberties[root_b] |= self.liberties[root_a]
            self.liberties[root_a] = 0
            # Increment rank of root_b only if ranks were equal
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_b] += 1
            self.rank[root_a] = -1

        new_root = self.find_no_compression(a)
        self.liberties[new_root] = calculate_group_liberties(self.stones[new_root], self.state, self.board_size)

    def undo_move_changes(self, undo_stack: list[UndoAction], zobrist: ZobristHash) -> None:
        """Apply the undo stack to revert changes to both state and UF"""
        for action in reversed(undo_stack):
            if action.action_type == UndoActionType.SET_STATE:
                x, y = (
                    action.position // self.board_size,
                    action.position % self.board_size,
                )
                # place: 0 -> 1, revert means, remove 1
                # capture: 1 -> 0, revert means, add 1
                self.hash = zobrist.update_hash(self.hash, action.position, self.state[x][y], action.value)
                self.state[x][y] = action.value
            else:
                # Let the UF handle other undo actions
                if action.action_type == UndoActionType.SET_PARENT:
                    self.parent[action.position] = action.value
                elif action.action_type == UndoActionType.SET_RANK:
                    self.rank[action.position] = action.value
                elif action.action_type == UndoActionType.SET_STONES:
                    self.stones[action.position] = action.value
                elif action.action_type == UndoActionType.SET_LIBERTIES:
                    self.liberties[action.position] = action.value
                elif action.action_type == UndoActionType.SET_COLOR:
                    self.colors[action.position] = action.value

    @staticmethod
    def get_uf_from_state(state: State, zobrist: ZobristHash | None) -> "UnionFind":

        width: int = state.shape[0]

        parent = np.full(width * width, -1, dtype=np.int8)
        colors = np.full(width * width, -1, dtype=np.int8)
        rank = np.full(width * width, -1, dtype=np.int8)
        stones = np.zeros(width * width, dtype=np.int64)
        liberties = np.zeros(width * width, dtype=np.int64)
        uf = UnionFind(state, parent, colors, rank, stones, liberties, width)

        if zobrist is None:
            uf.hash = np.uint64(0)
        else:
            uf.hash = zobrist.compute_hash(state)

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
                                add_stone_to_group(liberties, n_root, idx)
                    continue

                parent[idx] = idx
                colors[idx] = state[x][y]
                stones[idx] = 0
                add_stone_to_group(stones, idx, idx)
                liberties[idx] = 0
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
                            add_stone_to_group(liberties, idx_root, nidx)
                        # neighbor is same color
                        elif colors[nidx] == color:
                            uf.union(idx, nidx)
                            idx_root = uf.find(idx)
                        # neighbor is enemy
                        elif uf.colors[nidx] == enemy:
                            enemy_group = uf.find(nidx)
                            remove_stone_from_group(uf.liberties, enemy_group, idx)

        return uf

    def copy(self):
        uf = UnionFind(
            self.state.copy(),
            self.parent.copy(),
            self.colors.copy(),
            self.rank.copy(),
            self.stones.copy(),
            self.liberties.copy(),
            self.board_size,
        )
        uf.hash = self.hash
        return uf


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

        # self.state: State = state
        self.current_player = 1  # black starts
        self.komi = komi

        # union find data structure
        size = board_width * board_width
        parent = np.full(size, -1, dtype=np.int8)
        colors = np.full(size, -1, dtype=np.int8)
        rank = np.full(size, -1, dtype=np.int8)
        stones = np.zeros(size, dtype=np.int64)
        liberties = np.zeros(size, dtype=np.int64)

        self.zobrist = ZobristHash(self.board_width)
        self.uf: UnionFind = UnionFind(state, parent, colors, rank, stones, liberties, self.board_width)
        self.uf.hash = self.zobrist.compute_hash(state)
        self.hash_history: list[np.uint64] = [self.uf.hash]

    def __str__(self):
        board = self.decode_state(self.uf.state)
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

    def encode_state(self, state: list[str]) -> State:
        """
        Converts a list of strings (like [".....", "..X..", ...]) into a numpy array
        with the following encoding:
          '.' -> 0 (empty)
          'X' -> 1 (black)
          'O' -> 2 (white)
          '#' -> 3 (disabled)
        """
        transformed = np.zeros([self.board_width, self.board_width], dtype=np.int8)
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
        uf: UnionFind,
        additional_history: list[np.uint64] = [],
    ) -> UnionFind | None:
        if action == self.board_size:  # Pass move
            return uf

        color = 2 if is_white else 1
        new_uf = uf.copy()

        is_legal, _ = self.simulate_move(
            new_uf,
            action,
            color,
            additional_history,
        )

        if is_legal:
            return new_uf
        else:
            return None

    def flood_fill_territory(
        self, state: State, x: int, y: int, visited: np.ndarray[Any, np.dtype[np.bool_]]
    ) -> tuple[int | None, np.ndarray[Any, np.dtype[np.bool_]]]:

        queue: deque[tuple[int, int]] = deque()
        queue.append((x, y))

        territory: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros((self.board_width, self.board_height), dtype=np.bool_)
        adjacent_colors: set[int] = set()

        dx = np.array([0, 1, 0, -1])
        dy = np.array([1, 0, -1, 0])

        while queue:
            cx, cy = queue.popleft()
            if visited[cx, cy]:
                continue
            visited[cx, cy] = True

            # If its empty -> part of the territory
            if state[cx, cy] == 0:
                territory[cx, cy] = True

                nx = cx + dx
                ny = cy + dy
                valid_mask = (0 <= nx) & (nx < self.board_width) & (0 <= ny) & (ny < self.board_height)
                # Check pre-computed neighbors
                for i in range(4):
                    if not valid_mask[i]:
                        continue
                    neighbor_x = nx[i]
                    neighbor_y = ny[i]
                    cell_value = state[neighbor_x, neighbor_y]
                    if cell_value == 0:
                        if not visited[neighbor_x, neighbor_y]:
                            queue.append((neighbor_x, neighbor_y))
                    # If it its a stone save its color
                    elif cell_value in [1, 2]:
                        adjacent_colors.add(int(cell_value))

        # if territory is too large, its not valid
        if np.count_nonzero(territory) > 2 * self.board_width:
            return None, territory

        # If territory touches exactly one color, it belongs to that color
        if len(adjacent_colors) == 1:
            return adjacent_colors.pop(), territory
        # Otherwise its disputed or belongs to no one
        return None, territory

    def get_territory_scores(self, state: State) -> tuple[int, int]:
        """
        Finds how many empty intersections belong to black or white via territory.
        Returns (white_territory, black_territory).
        """
        visited: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros((self.board_width, self.board_height), dtype=np.bool_)
        white_territory = 0
        black_territory = 0

        for x in range(self.board_width):
            for y in range(self.board_height):
                # 0 means node is empty
                if not visited[x, y] and state[x, y] == 0:
                    color_owner, territory = self.flood_fill_territory(state, x, y, visited)
                    if color_owner == 1:  # black
                        black_territory += np.count_nonzero(territory)
                    elif color_owner == 2:  # white
                        white_territory += np.count_nonzero(territory)

        return white_territory, black_territory

    def get_score(self, uf: UnionFind, komi: float) -> dict[str, dict[str, float]]:
        """
        Computes the score for white and black, including komi for white.
        - Each stone on the board counts as 1 point
        - Each empty node fully surrounded by one color also counts as territory
        """
        # Count stones directly
        black_pieces = np.sum(uf.state == 1)
        white_pieces = np.sum(uf.state == 2)

        # Get territory counts
        white_territory, black_territory = self.get_territory_scores(uf.state)

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

    def get_liberties(self, state: State, x: int, y: int, visited: set[tuple[int, int]]):
        """
        Returns a set of all liberties the color at (x, y) has and the territory
        """
        color: int = state[x][y]
        if color in [0, 3]:
            raise ValueError("Empty Cell or Disabled Cell cannot have liberties")

        queue: deque[tuple[int, int]] = deque()
        queue.append((x, y))

        stones: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            # If same color -> stone of the group
            if state[cx][cy] == color:
                stones.add((cx, cy))
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

        return liberties, stones

    def simulate_move(
        self,
        uf: UnionFind,
        action: int,
        color: int,
        additional_history: list[np.uint64] = [],
    ) -> tuple[bool, list[UndoAction]]:
        if action == self.board_height * self.board_height:
            return True, []

        x, y = self.decode_action(action)
        if uf.state[x][y] != 0:
            return False, []

        undo_stack: list[UndoAction] = []

        undo_stack.append(UndoAction(UndoActionType.SET_STATE, action, uf.state[x][y]))
        uf.hash = self.zobrist.add_stone(uf.hash, action, color)
        uf.state[x][y] = color

        # color = 1 => enemy = 2; color = 2 => ememy = 1
        enemy = 3 - color

        # Record original UF state for this position
        undo_stack.append(UndoAction(UndoActionType.SET_PARENT, action, uf.parent[action]))
        undo_stack.append(UndoAction(UndoActionType.SET_COLOR, action, uf.colors[action]))
        undo_stack.append(UndoAction(UndoActionType.SET_RANK, action, uf.rank[action]))
        undo_stack.append(UndoAction(UndoActionType.SET_STONES, action, uf.stones[action]))
        undo_stack.append(UndoAction(UndoActionType.SET_LIBERTIES, action, uf.liberties[action]))

        # Initialize the new stone's data
        uf.parent[action] = action  # stone is its own root initially
        uf.colors[action] = color
        uf.stones[action] = 0
        add_stone_to_group(uf.stones, action, action)
        uf.liberties[action] = 0  # liberties are bitmask, 0 (empty) initially
        uf.rank[action] = 0  # new single nodes start with rank 0

        action_root = action
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                nidx = self.encode_action(nx, ny)
                # neighbor is empty
                if uf.state[nx][ny] == 0:
                    undo_stack.append(UndoAction(UndoActionType.SET_LIBERTIES, action_root, uf.liberties[action_root]))
                    add_stone_to_group(uf.liberties, action_root, nidx)
                # neighbor is same color
                elif uf.colors[nidx] == color:
                    root_nidx = uf.find_no_compression(nidx)

                    # Record original parent, rank, stones, liberties for both roots
                    undo_stack.append(UndoAction(UndoActionType.SET_PARENT, action_root, uf.parent[action_root]))
                    undo_stack.append(UndoAction(UndoActionType.SET_RANK, action_root, uf.rank[action_root]))
                    undo_stack.append(UndoAction(UndoActionType.SET_STONES, action_root, uf.stones[action_root]))
                    undo_stack.append(UndoAction(UndoActionType.SET_LIBERTIES, action_root, uf.liberties[action_root]))

                    undo_stack.append(UndoAction(UndoActionType.SET_PARENT, root_nidx, uf.parent[root_nidx]))
                    undo_stack.append(UndoAction(UndoActionType.SET_RANK, root_nidx, uf.rank[root_nidx]))
                    undo_stack.append(UndoAction(UndoActionType.SET_STONES, root_nidx, uf.stones[root_nidx]))
                    undo_stack.append(UndoAction(UndoActionType.SET_LIBERTIES, root_nidx, uf.liberties[root_nidx]))

                    uf.union(action, nidx)
                    action_root = uf.find_no_compression(action)
                # neighbor is enemy
                elif uf.colors[nidx] == enemy:
                    enemy_group = uf.find_no_compression(nidx)

                    # Record original liberties for the enemy group
                    undo_stack.append(UndoAction(UndoActionType.SET_LIBERTIES, enemy_group, uf.liberties[enemy_group]))
                    remove_stone_from_group(uf.liberties, enemy_group, action)

                    # Recompute liberties for the enemy group in case
                    # TODO: capsule in a function for less code duplication
                    uf.liberties[enemy_group] = calculate_group_liberties(
                        uf.stones[enemy_group], uf.state, self.board_width
                    )

                    # capture enemy group if no liberties
                    if uf.liberties[enemy_group] == 0:
                        stone_indices = get_bit_indices(uf.stones[enemy_group])
                        # Record original state for the enemy group
                        undo_stack.append(UndoAction(UndoActionType.SET_STONES, enemy_group, uf.stones[enemy_group]))

                        for stone in stone_indices:
                            sx, sy = self.decode_action(stone)
                            # Record original state value
                            undo_stack.append(UndoAction(UndoActionType.SET_STATE, stone, uf.state[sx][sy]))
                            uf.hash = self.zobrist.remove_stone(uf.hash, stone, uf.state[sx][sy])
                            uf.state[sx][sy] = 0

                            # Record original stone properties
                            undo_stack.append(UndoAction(UndoActionType.SET_COLOR, stone, uf.colors[stone]))
                            undo_stack.append(UndoAction(UndoActionType.SET_PARENT, stone, uf.parent[stone]))
                            undo_stack.append(UndoAction(UndoActionType.SET_RANK, stone, uf.rank[stone]))

                            uf.colors[stone] = -1
                            uf.parent[stone] = -1
                            uf.rank[stone] = -1
                        uf.stones[enemy_group] = 0
                        uf.liberties[enemy_group] = 0

                        # update liberties of neighboring groups
                        for stone in stone_indices:
                            sx, sy = self.decode_action(stone)
                            for ddx, ddy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                snx, sny = sx + ddx, sy + ddy
                                if 0 <= snx < self.board_width and 0 <= sny < self.board_height:
                                    snidx = self.encode_action(snx, sny)
                                    if uf.colors[snidx] == color:
                                        neighbor_root = uf.find_no_compression(snidx)
                                        # Record original liberties
                                        if not any(
                                            a.action_type == UndoActionType.SET_LIBERTIES
                                            and a.position == neighbor_root
                                            for a in undo_stack
                                        ):
                                            undo_stack.append(
                                                UndoAction(
                                                    UndoActionType.SET_LIBERTIES,
                                                    neighbor_root,
                                                    uf.liberties[neighbor_root],
                                                )
                                            )
                                        add_stone_to_group(uf.liberties, neighbor_root, stone)

        # check if placed stone has liberties
        if uf.liberties[action_root] == 0:
            # move was actually a suicide
            uf.undo_move_changes(undo_stack, self.zobrist)
            return False, undo_stack

        # check for repeat
        is_repeat = self.check_state_is_repeat(uf.hash, additional_history)
        if is_repeat:
            # move was actually a repeat
            uf.undo_move_changes(undo_stack, self.zobrist)
            return False, undo_stack

        return True, undo_stack

    def verify_uf_consistency(self, state: State, uf: UnionFind) -> bool:
        """Verify that the UnionFind structure is consistent with the board state"""
        for x in range(self.board_width):
            for y in range(self.board_height):
                idx = self.encode_action(x, y)
                root = uf.find_no_compression(idx)
                if state[x][y] in [1, 2]:  # Stone exists
                    if uf.colors[idx] != state[x][y]:
                        return False
                    elif uf.parent[idx] == -1:
                        return False
                    elif uf.rank[root] == -1:
                        return False
                    elif uf.stones[root] == 0:
                        return False
                else:  # Empty or disabled
                    if uf.colors[idx] != -1:
                        return False
                    elif uf.rank[idx] != -1:
                        return False

        # check parent consistency
        for x in range(self.board_width):
            for y in range(self.board_height):
                if state[x][y] not in [1, 2]:
                    continue
                idx = self.encode_action(x, y)
                root = uf.find_no_compression(idx)
                root_coord = self.decode_action(root)
                # check if root is actually parent of idx
                _, stones = self.get_liberties(state, x, y, set())
                if root_coord not in stones:
                    print(f"root not in stones: {root_coord} {stones}")
                    return False

        return True

    def check_state_is_repeat(self, hash: np.uint64, additional_history: list[np.uint64] = []) -> bool:
        in_history = hash in self.hash_history
        in_additional = hash in additional_history

        return in_history or in_additional

    def make_move(self, action: int, is_white: bool) -> tuple[UnionFind, int, bool]:
        uf_after_move = self.state_after_action(action, is_white, self.uf)
        assert uf_after_move is not None, "Illegal move"
        game_ended = self.has_game_ended(action, is_white, self.uf)

        self.uf = uf_after_move
        self.current_player = 3 - self.current_player

        # update history and prev action
        self.history.append(self.uf.state.copy())
        self.hash_history.append(self.uf.hash)
        self.previous_action = action

        # outcome is 1 black won, -1 white won, 0 not ended
        outcome = 0
        if game_ended:
            score = self.get_score(self.uf, self.komi)
            print(f"score: {score}")
            outcome = 1 if score["black"]["sum"] > score["white"]["sum"] else -1

        return self.uf, outcome, game_ended

    def get_valid_moves(
        self,
        uf: UnionFind,
        is_white: bool,
        history: list[np.uint64] = [],
    ) -> np.ndarray[Any, np.dtype[np.bool_]]:
        player = 2 if is_white else 1

        # if len(history) == 0:
        #     history = self.hash_history

        legal_moves: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros_like(uf.state, dtype=bool)
        empty_mask = uf.state == 0
        empty_positions = np.where(empty_mask)
        for x, y in zip(empty_positions[0], empty_positions[1]):
            action = self.encode_action(int(x), int(y))
            is_legal, undo = self.simulate_move(
                uf,
                action,
                player,
                history,
            )
            if is_legal:  # only needs to undo if legal, since illegal moves are not persisted
                uf.undo_move_changes(undo, self.zobrist)
                legal_moves[x][y] = True

        return legal_moves

    def has_game_ended(
        self,
        action: int,
        is_white: bool,
        uf: UnionFind,
        additional_history: list[np.uint64] = [],
    ) -> bool:
        # double pass
        if self.previous_action == action == self.board_width * self.board_height:
            return True

        # previous pass, current has no valid moves
        valid = self.get_valid_moves(uf, is_white, additional_history)
        if self.previous_action == self.board_width * self.board_height and np.sum(valid) == 0:
            return True

        # board is full
        has_empty_node = np.any(uf.state == 0)
        if has_empty_node:
            return False

        return False

    def get_history(self) -> list[State]:
        return self.history

    def get_hash_history(self) -> list[np.uint64]:
        return self.hash_history


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
    go = Go_uf(5, decoded_board, 5.5)
    print(go.get_liberties(decoded_board, 0, 2, set()))

    # go = Go_uf(5, decoded_board, 5.5)

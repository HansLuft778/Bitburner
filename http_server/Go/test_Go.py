import unittest
from Go_uf import Go_uf, UnionFind
import numpy as np

# from go_types import State
from typing import Any

State = np.ndarray[Any, np.dtype[np.int8]]


def encode_state(state: list[str]) -> State:
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


class TestGo(unittest.TestCase):
    def setUp(self):
        self.test_boards: list[tuple[list[str], dict[str, float]]] = [
            (
                [".X.OO", "O..XX", ".OX.#", ".OOX.", "O.OX."],
                {"black": 9, "white": 16.5},
            ),
            (["..X..", ".....", "#OO.X", ".XO..", "X...."], {"black": 5, "white": 8.5}),
            (
                ["..##O", "#XXX.", ".XO.O", ".XXX.", "O..OO"],
                {"black": 9, "white": 11.5},
            ),
            (
                ["..#O.", ".XOOO", "XOO.X", ".X..X", "...X."],
                {"black": 10, "white": 12.5},
            ),
            (
                [".XX..", "O.OXX", ".O.O#", ".OOX.", "O.OX."],
                {"black": 10, "white": 17.5},
            ),
        ]

    def test_get_score(self):
        for board, expected in self.test_boards:
            enc = encode_state(board)
            go = Go_uf(5, enc, 5.5, False)
            score = go.get_score(go.uf)
            self.assertEqual(score["black"]["sum"], expected["black"])
            self.assertEqual(score["white"]["sum"], expected["white"])

    def test_score_components(self):
        for board, _ in self.test_boards:
            enc = encode_state(board)
            go = Go_uf(5, enc, 5.5, False)
            score = go.get_score(go.uf)

            # Test that components add up correctly
            black_sum = score["black"]["pieces"] + score["black"]["territory"]
            white_sum = score["white"]["pieces"] + score["white"]["territory"] + score["white"]["komi"]

            self.assertEqual(score["black"]["sum"], black_sum)
            self.assertEqual(score["white"]["sum"], white_sum)

            # Test komi values
            self.assertEqual(score["black"]["komi"], 0)
            self.assertEqual(score["white"]["komi"], 5.5)

    def place_stone_edge_case_7x7(self):
        # fmt: off
        parent= np.array([-1, -1, 8, 22, -1, 5, -1, 22, 8, 8, 22, 22, -1, -1, 22, 22, 22, 24, 19, 22, -1, -1, 22, 22, 22, 33, 33, -1, 29, 29, 29, 33, 33, 29, 33, -1, 36, 29, -1, 33, 33, 29, -1, -1, 44, -1, -1, -1, -1])
        colors= np.array([-1, -1, 2, 1, -1, 1, -1, 1, 2, 2, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 2, 2, -1, 2, 2, 2, 2, 2, 2, 2, -1, 1, 2, -1, 2, 2, 2, -1, -1, 1, -1, -1, -1, -1])
        rank= np.array([-1, -1, -1, -1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1])
        stones= np.array([0, 0, 0, 0, 0, 32, 0, 0, 772, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30395528, 0, 0, 0, 0, 0, 0, 4019921616896, 0, 0, 0, 0, 0, 0, 68719476736, 0, 0, 0, 0, 0, 0, 0, 17592186044416, 0, 0, 0, 0])
        liberties= np.array([0, 0, 0, 0, 0, 4176, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2101264, 0, 0, 0, 0, 0, 0, 351878216941568, 0, 0, 0, 0, 0, 0, 8830452760576, 0, 0, 0, 0, 0, 0, 0, 8796093022208, 0, 0, 0, 0])
        # fmt: on

        decoded_board = np.array(
            [
                [3, 0, 2, 1, 0, 1, 0],
                [1, 2, 2, 1, 1, 0, 3],
                [1, 1, 1, 1, 1, 1, 3],
                [0, 1, 1, 1, 2, 2, 0],
                [2, 2, 2, 2, 2, 2, 2],
                [0, 1, 2, 3, 2, 2, 2],
                [3, 0, 1, 3, 0, 3, 0],
            ],
            dtype=np.int8,
        )

        uf = UnionFind(decoded_board, parent, colors, rank, stones, liberties, 7)
        go = Go_uf(7, decoded_board, 5.5, False)
        go.uf = uf
        out_uf, outcome, has_ended = go.make_move(43, True)
        self.assertIsNotNone(out_uf)
        self.assertEqual(outcome, 0)
        self.assertEqual(has_ended, False)


if __name__ == "__main__":
    unittest.main()

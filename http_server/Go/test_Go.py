import unittest
from Go_uf import Go_uf
import numpy as np
from go_types import State


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


if __name__ == "__main__":
    unittest.main()

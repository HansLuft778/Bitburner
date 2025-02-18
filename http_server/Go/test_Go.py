import unittest
from Go import Go


class TestGo(unittest.TestCase):
    def setUp(self):
        self.test_boards = [
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
            go = Go(5, 5, board)
            score = go.get_score(5.5)
            self.assertAlmostEqual(score["black"]["sum"], expected["black"])
            self.assertAlmostEqual(score["white"]["sum"], expected["white"])

    def test_score_components(self):
        for board, _ in self.test_boards:
            go = Go(5, 5, board)
            score = go.get_score(5.5)

            # Test that components add up correctly
            black_sum = score["black"]["pieces"] + score["black"]["territory"]
            white_sum = (
                score["white"]["pieces"]
                + score["white"]["territory"]
                + score["white"]["komi"]
            )

            self.assertEqual(score["black"]["sum"], black_sum)
            self.assertEqual(score["white"]["sum"], white_sum)

            # Test komi values
            self.assertEqual(score["black"]["komi"], 0)
            self.assertEqual(score["white"]["komi"], 5.5)

    def test_move_validation(self):
        # Test case 1: Board with open center
        board1 = [
            ".OX.O",
            ".OXOX",
            "..O..",
            ".....",
            ".XXXO",
        ]
        go = Go(5, 5, board1)

        # Position (0,3) should be legal for both players
        go.current_player = 1  # Black
        self.assertTrue(go.evaluateMoveIsVaid(go.state, 0, 3))
        go.current_player = 2  # White
        self.assertTrue(go.evaluateMoveIsVaid(go.state, 0, 3))

        # Test case 2: Board with edge constraints
        board2 = ["#XO.X", "#XOXX", "#.XOO", "#OO.O", "#...X"]
        go = Go(5, 5, board2)

        # Position (0,3) should be legal for both players
        go.current_player = 1  # Black
        self.assertTrue(go.evaluateMoveIsVaid(go.state, 0, 3))
        go.current_player = 2  # White
        self.assertTrue(go.evaluateMoveIsVaid(go.state, 0, 3))

        # Position (2,1) should only be legal for white
        go.current_player = 1  # Black
        self.assertFalse(go.evaluateMoveIsVaid(go.state, 2, 1))
        go.current_player = 2  # White
        self.assertTrue(go.evaluateMoveIsVaid(go.state, 2, 1))


if __name__ == "__main__":
    unittest.main()

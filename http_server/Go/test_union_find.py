import unittest
from Go.Go_uf import UnionFind
import numpy as np


class TestUnionFind(unittest.TestCase):

    def setUp(self):
        return super().setUp()

    def test_union_find(self):
        decoded_board = np.array(
            [
                [3, 1, 0, 0, 0],
                [2, 1, 0, 0, 2],
                [2, 0, 1, 1, 2],
                [3, 1, 0, 0, 2],
                [0, 1, 0, 0, 2],
            ]
        )

        uf = UnionFind.get_uf_from_state(decoded_board)
        # fmt: off
        should_parent = np.array(
            [
                [-1,  1, -1, -1, -1],
                [ 5,  1, -1, -1,  9],
                [ 5, -1, 12, 12,  9],
                [-1, 16, -1, -1,  9],
                [-1, 16, -1, -1,  9],
            ]
        )
        should_color = np.array(
            [
                [-1,  1, -1, -1, -1],
                [ 2,  1, -1, -1,  2],
                [ 2, -1,  1,  1,  2],
                [-1,  1, -1, -1,  2],
                [-1,  1, -1, -1,  2],
            ]
        )
        should_rank = np.array(
            [
                [-1,  1, -1, -1, -1],
                [ 1, -1, -1, -1,  1],
                [-1, -1,  1, -1, -1],
                [-1,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ]
        )
        should_stones = [set(), {1, 6}, set(), set(), set(), 
                         {10, 5}, set(), set(), set(), {24, 9, 19, 14}, 
                         set(), set(), {12, 13}, set(), set(), 
                         set(), {16, 21}, set(), set(), set(), 
                         set(), set(), set(), set(), set()]

        should_liberties = [set(), {2, 11, 7}, set(), set(), set(), 
                            {11}, set(), set(), set(), {8, 18, 4, 23}, 
                            set(), set(), {7, 8, 11, 17, 18}, set(), set(), 
                            set(), {17, 11, 20, 22}, set(), set(), set(), 
                            set(), set(), set(), set(), set()]
        # fmt: on

        # assert
        self.assertTrue(np.array_equal(uf.parent, should_parent.reshape(25)), uf.parent)
        self.assertTrue(np.array_equal(uf.colors, should_color.reshape(25)), uf.colors)
        self.assertTrue(np.array_equal(uf.rank, should_rank.reshape(25)), uf.rank)
        self.assertTrue(uf.stones == should_stones, uf.stones)
        self.assertTrue(uf.liberties == should_liberties, uf.liberties)

    def test_union_find_2(self):
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
        # fmt: off
        should_parent = np.array(
            [
                [-1, -1,  6,  3,  3],
                [-1,  6,  6,  3, -1],
                [-1, 11, -1, 13, 14],
                [15, -1, -1, 18, 19],
                [-1, -1, -1, -1, -1],
            ]
        )
        should_color = np.array(
            [
                [-1, -1,  1,  2,  2],
                [-1,  1,  1,  2, -1],
                [-1,  2, -1,  1,  2],
                [ 1, -1, -1,  2,  1],
                [-1, -1, -1, -1, -1],
            ]
        )
        should_rank = np.array(
            [
                [-1, -1, -1,  1, -1],
                [-1,  1, -1, -1, -1],
                [-1,  0, -1,  0,  0],
                [ 0, -1, -1,  0,  0],
                [-1, -1, -1, -1, -1],
            ]
        )
        should_stones = [set(), set(), set(), {8, 3, 4}, set(), 
                         set(), {2, 6, 7}, set(), set(), set(), 
                         set(), {11}, set(), {13}, {14}, 
                         {15}, set(), set(), {18}, {19}, 
                         set(), set(), set(), set(), set()]
        
        should_liberties = [set(), set(), set(), {9}, set(), 
                            set(), {1, 12, 5}, set(), set(), set(), 
                            set(), {16, 10, 12}, set(), {12}, {9},
                            {16, 10, 20}, set(), set(), {17, 23}, {24}, 
                            set(), set(), set(), set(), set()]
        # fmt: on

        # assert
        self.assertTrue(np.array_equal(uf.parent, should_parent.reshape(25)), uf.parent)
        self.assertTrue(np.array_equal(uf.colors, should_color.reshape(25)), uf.colors)
        self.assertTrue(np.array_equal(uf.rank, should_rank.reshape(25)), uf.rank)
        self.assertTrue(uf.stones == should_stones, uf.stones)
        self.assertTrue(uf.liberties == should_liberties, uf.liberties)


if __name__ == "__main__":
    unittest.main()

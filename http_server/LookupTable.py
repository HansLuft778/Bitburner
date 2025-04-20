from typing import Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class LookupTable:
    def __init__(
        self,
        board_width: int,
        c_score: float
    ):
        self.board_width = board_width
        self.c_score = c_score

        max_sigma = 30
        min_mu = -30.5
        max_mu = 30.5
        max_p1 = (min_mu - max_mu) / board_width

        print("creating p2 range...")
        self.p2_range = np.arange(0, max_sigma / board_width, 0.1)
        print("creating p1 range...")
        self.p1_range = np.arange(max_p1, -max_p1, 0.1)

        print("creating table...")
        self.table = np.zeros((self.p1_range.shape[0], self.p2_range.shape[0]))

        self.create_table()
        print("Table precomputation done!")

        self.interpolator = RegularGridInterpolator((self.p1_range, self.p2_range), self.table)

    def _u_score_normalized(
        self, x: np.ndarray[Any, np.dtype[np.float32]], p1: float, p2: float, c_score: float
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return c_score * (2 / np.pi) * np.arctan(p2 * x + p1)

    def normal_pdf(
        self, x: np.ndarray[Any, np.dtype[np.float32]], mu: float, sigma: float
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        sigma_squared = sigma * sigma
        return (np.e ** -((x - mu) ** 2 / (2.0 * sigma_squared))) / np.sqrt(2.0 * np.pi * sigma_squared)

    def create_table(self):
        step = 1

        for i, p1 in enumerate(self.p1_range):
            for j, p2 in enumerate(self.p2_range):
                xs = np.arange(-30.5, 30.5 + step, step)
                ys_1 = self._u_score_normalized(xs, p1, p2, self.c_score)
                ys_2 = self.normal_pdf(xs, 0, 1)

                self.table[i, j] = np.trapezoid(ys_1 * ys_2, xs)

    def get_expected_uscore(self, mu: float, sigma: float, x_0: float):
        p1 = (mu - x_0) / self.board_width
        p2 = sigma / self.board_width

        return self.interpolator((p1, p2))

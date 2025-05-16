# type: ignore
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from collections import deque, defaultdict
from datetime import datetime
import math
import os
from zipfile import ZipFile
from typing import TYPE_CHECKING

from Go.Go_uf import UnionFind
from go_types import State

if TYPE_CHECKING:
    from gameserver_local_uf import GameServerGo


def cleanup_out_folder(out_folder: str):
    """Zips every png file from the out folder into a zip and removes them from the folder.
    Args:
        out_folder (str): Absolute path to the folder containing the png files.
    """
    content = os.listdir(out_folder)

    zips = [f for f in content if f.endswith(".zip")]
    run_id = (
        int(max(zips, key=lambda x: int(x.split("_")[1].split(".")[0])).split("_")[1].split(".")[0]) if zips else -1
    )

    images = [f for f in content if f.endswith(".png")]
    if len(images) == 0:
        return

    zip_path = os.path.join(out_folder, f"run_{run_id + 1}.zip")
    with ZipFile(zip_path, "w") as myzip:
        for img in images:
            full_path = os.path.join(out_folder, img)
            myzip.write(full_path, arcname=img)
            os.remove(full_path)


class Plot:
    def __init__(self, ax, title, xlabel, ylabel, label=None, maxlen=None, color=None):
        self.data = [] if maxlen is None else deque(maxlen=maxlen)
        self.ax = ax
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        (self.line,) = self.ax.plot([], label=label, color=color)
        if label:
            self.ax.legend()

    def update(self, new_value, cumulative=False, max_size=2000):
        if cumulative:
            if self.data:
                self.data.append(new_value + self.data[-1])
            else:
                self.data.append(new_value)
        else:
            self.data.append(float(new_value))

        # Downsample if needed
        if len(self.data) > max_size:
            self.data = list(self.data)[::2]

        self.line.set_ydata(self.data)
        self.line.set_xdata(range(len(self.data)))
        self.ax.relim()
        self.ax.autoscale_view()


class Plotter:
    def __init__(self, rows=2, cols=2, window_title: str = "Go Bot Losses"):
        plt.ion()  # Enable interactive mode

        # Create a configurable subplot layout
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(window_title)
        # Convert to 2D array for consistent indexing
        if rows == 1 and cols == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1:
            self.axes = np.array([self.axes])
        elif cols == 1:
            self.axes = np.array([[ax] for ax in self.axes])

        # Flatten for easy access
        self.axes_flat = self.axes.flatten()

        # Dictionary to store all plots
        self.plots = {}

        # Legacy attributes for backward compatibility
        self.ax1 = self.axes_flat[0] if len(self.axes_flat) > 0 else None
        self.ax2 = self.axes_flat[1] if len(self.axes_flat) > 1 else None
        self.ax3 = self.axes_flat[2] if len(self.axes_flat) > 2 else None
        self.ax4 = self.axes_flat[3] if len(self.axes_flat) > 3 else None

        plt.tight_layout()

    def add_plot(self, name, ax, title, xlabel, ylabel, label=None, maxlen=None, color=None):
        """Add a new plot to be tracked"""
        self.plots[name] = Plot(ax, title, xlabel, ylabel, label, maxlen, color)
        return self.plots[name]

    def update_stat(self, name, value, cumulative=False, max_size=2000, draw=True):
        """Generic method to update any stat"""
        if name not in self.plots:
            raise ValueError(f"Plot '{name}' not found. Add it first with add_plot().")

        self.plots[name].update(value, cumulative, max_size)
        if draw:
            self.draw_and_flush()

    def draw_and_flush(self):
        """Draw and flush all plot updates at once"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.tight_layout()

    # Legacy methods for backward compatibility
    def update_wins_black(self, new_reward: float, draw=True):
        self.update_stat("cumulative_reward_black", new_reward, cumulative=True, max_size=200, draw=draw)

    def update_wins_white(self, new_reward: float, draw=True):
        self.update_stat("cumulative_reward_white", new_reward, cumulative=True, max_size=200, draw=draw)

    def update_loss(self, new_loss: float, draw=True):
        self.update_stat("loss", new_loss, draw=draw)

    def update_policy_loss(self, new_policy_loss: float, draw=True):
        self.update_stat("policy_loss_own", new_policy_loss, draw=draw)

    def update_value_loss(self, new_value_loss: float, draw=True):
        self.update_stat("value_loss", new_value_loss, draw=draw)


class ModelOverlay:
    def __init__(self, board_size: int, komi: float) -> None:
        self.board_size = board_size
        max_score = board_size * board_size + komi
        self.num_scores = int(max_score * 2 + 1)
        self.possible_scores = np.linspace(-max_score, max_score, self.num_scores)

    def heatmap(
        self,
        uf: UnionFind,
        uf_after: UnionFind | None,
        uf_after_after: UnionFind | None,
        model_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        is_white: bool,
        server: "GameServerGo",
        final_score: float,
        save_image_to_disk: bool = False,
        save_path: str = "model_overlay.png",
    ) -> Figure:
        # If we're saving, switch to a non-interactive backend temporarily
        if save_image_to_disk:
            current_backend = plt.get_backend()
            plt.switch_backend("Agg")

        pi, pi_opp, game_outcome_tensor, ownership, score_logits, v_pooled = model_output

        win_prob_pred = game_outcome_tensor[0][0].item()
        mu_score = game_outcome_tensor[0][1].item()
        sigma_score = game_outcome_tensor[0][2].item()
        score = server.go.get_score(uf)

        score_diff = score["black"]["sum"] - score["white"]["sum"]
        score_normalized = -score_diff if is_white else score_diff
        final_score_normalized = -final_score if is_white else final_score

        fig, ax = plt.subplots(3, 2, figsize=(8, 8))
        fig.suptitle(
            f"{save_path}\n"
            f"Move Prediction for {'White' if is_white else 'Black'} with current score: {score_normalized} | "
            f"final score: {final_score_normalized}\n"
            f"Outcome Prediction: value: {win_prob_pred:.2f}, {r'$\mu$'}: {mu_score:.2f}, {r'$\sigma$'}: {sigma_score:.2f}"
        )

        # ownership prediction plot
        im = ax[0][0].imshow(
            ownership.detach().cpu().squeeze().numpy(), cmap="PuOr", interpolation="nearest", vmin=-1, vmax=1
        )
        black_y, black_x = np.where(uf.state == 1)
        white_y, white_x = np.where(uf.state == 2)
        disabled_y, disabled_x = np.where(uf.state == 3)
        stone_size = 1000 / self.board_size  # Scale stone size based on board size
        ax[0][0].scatter(black_x, black_y, c="black", s=stone_size, alpha=1)
        ax[0][0].scatter(white_x, white_y, c="white", s=stone_size, alpha=1)
        ax[0][0].scatter(disabled_x, disabled_y, c="#808080", s=stone_size, alpha=1, marker="x")
        ax[0][0].set_title("Ownership prediction")
        fig.colorbar(im, ax=ax[0][0])

        # score cdf plot
        score_onehot = torch.zeros(self.num_scores)
        score_idx = int(self.num_scores / 2) + math.floor(final_score_normalized)
        score_onehot[score_idx] = 1.0

        target_cdf = torch.cumsum(score_onehot, dim=0)

        score_probs = F.softmax(score_logits.squeeze(), dim=0)
        predicted_cdf: torch.Tensor = torch.cumsum(score_probs, dim=0)

        ax[0][1].plot(self.possible_scores, predicted_cdf.detach().cpu().numpy(), label="Predicted CDF")
        ax[0][1].plot(self.possible_scores, target_cdf.numpy(), label="Target CDF")
        ax[0][1].set_title("score CDF prediction")
        ax[0][1].legend()

        # score pdf plot
        ax[1][1].bar(
            self.possible_scores, score_probs.detach().cpu().numpy(), label="Predicted PDF", alpha=0.5, width=0.8
        )

        ax[1][1].vlines(mu_score, 0, max(score_probs).item(), "red", "dashed", label=r"$\mu$")
        ax[1][1].set_title("score PDF prediction")
        ax[1][1].legend()

        pi_props = torch.softmax(pi.squeeze(), dim=0).detach().cpu().squeeze().numpy()
        pi_opp_props = torch.softmax(pi_opp.squeeze(), dim=0).detach().cpu().squeeze().numpy()

        pi_board = pi_props[:-1].reshape((self.board_size, self.board_size))
        pi_pass = pi_props[-1].item()
        pi_opp_board = pi_opp_props[:-1].reshape((self.board_size, self.board_size))
        pi_opp_pass = pi_opp_props[-1]

        # own move policy plot
        move_im = ax[1][0].imshow(pi_board, cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
        ax[1][0].set_title(f"Own Move Policy, pass: {pi_pass:.2f}")
        fig.colorbar(move_im, ax=ax[1][0])

        # opponent move policy plot
        move_im = ax[2][0].imshow(pi_opp_board, cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
        ax[2][0].set_title(f"Opp Move Policy, pass: {pi_opp_pass:.2f}")
        fig.colorbar(move_im, ax=ax[2][0])

        if uf_after is not None and uf_after_after is not None:
            next_state = uf_after.state
            next_next_state = uf_after_after.state
            diff = next_next_state - next_state
            placed_stone_y, placed_stone_x = np.where(diff > 0)
            removed_stones_y, removed_stones_x = np.where(diff < 0)
            assert (
                len(placed_stone_x) == len(placed_stone_y) and len(placed_stone_x) <= 1
            ), "There should be only one or none stone placed"
            # plot placed stone
            color = "black" if is_white else "white"
            ax[2][0].scatter(placed_stone_x, placed_stone_y, c=color, s=200, alpha=1)

        fig.tight_layout()

        if save_image_to_disk:
            plt.savefig(f"out/{save_path}")
            # plt.close(fig)  # Close the figure to prevent display
            plt.switch_backend(current_backend)  # Restore original backend

        return fig


class GameStatePlotter:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board_color = "#DEB887"
        self.line_color = "#5C4033"

        self.stone_size = 1000
        self.disabled_size = 350
        self.territory_size = 200

        plt.ion()

        self.fig, self.axes = plt.subplots(figsize=(self.board_size, self.board_size))

        self._setup_board()

    def _setup_board(self):
        """Sets up the static appearance of the board (background, grid, limits)."""
        self.axes.set_facecolor(self.board_color)

        # Set plot limits slightly outside the grid
        margin = 0.5
        self.axes.set_xlim(-margin, self.board_size - 1 + margin)
        self.axes.set_ylim(-margin, self.board_size - 1 + margin)

        # Draw grid lines
        for i in range(self.board_size):
            self.axes.axhline(i, color=self.line_color, lw=1, zorder=0)
            self.axes.axvline(i, color=self.line_color, lw=1, zorder=0)

        # Show axes with zero-based ticks
        self.axes.set_xticks(range(self.board_size))
        self.axes.set_yticks(range(self.board_size))
        # Hide the top and right spines for a cleaner look
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)

        # Ensure aspect ratio is equal
        self.axes.set_aspect("equal", adjustable="box")

    def plot(self, state: State, is_white: bool, scores: dict[str, dict[str, float | State]]):
        """Plots the current game state."""
        self.axes.cla()
        self._setup_board()

        black_y, black_x = np.where(state == 1)
        white_y, white_x = np.where(state == 2)
        disabled_y, disabled_x = np.where(state == 3)

        score_black: float = scores["black"]["sum"]
        score_white: float = scores["white"]["sum"]

        black_terr_y, black_terr_x = np.where(scores["black"]["territory"] == 1)
        white_terr_y, white_terr_x = np.where(scores["white"]["territory"] == 1)

        self.axes.set_title(
            f"{"White" if is_white else "Black"}\nScore Black: {score_black} Score White: {score_white}"
        )
        self.axes.scatter(black_x, black_y, c="black", s=self.stone_size, alpha=1, zorder=1, edgecolors="dimgray")
        self.axes.scatter(white_x, white_y, c="white", s=self.stone_size, alpha=1, zorder=1, edgecolors="darkgray")

        # territory
        self.axes.scatter(
            black_terr_x,
            black_terr_y,
            c="dimgray",  # Dark gray for black territory
            s=self.territory_size,
            marker="s",  # Square marker
            alpha=0.6,  # Slightly transparent
            zorder=0.5,  # Between grid (0) and stones (1)
        )
        self.axes.scatter(
            white_terr_x,
            white_terr_y,
            c="lightgray",  # Light gray for white territory
            s=self.territory_size,
            marker="s",  # Square marker
            alpha=0.6,  # Slightly transparent
            zorder=0.5,  # Between grid (0) and stones (1)
        )

        # disabled
        self.axes.scatter(
            disabled_x,
            disabled_y,
            c="#606060",
            s=self.disabled_size,
            alpha=1,
            marker="x",
            zorder=1,
            lw=2,
        )

        self.fig.tight_layout(pad=0.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class TensorBoardPlotter:
    def __init__(self, comment: str = ""):
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=f"runs/Go_bot_{current_date}", comment=comment)

        self.global_step: dict[str, int] = defaultdict(int)
        self.cumulative_values: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update_stat(self, plot_id: str, value: float | int):
        gloabal_step = self.global_step.get(plot_id, 0)
        self.writer.add_scalar(plot_id, value, gloabal_step)
        self.global_step[plot_id] += 1

    def update_stat_dict(self, plot_id: str, value: dict[str, int], cumulative: bool = False):
        value_to_log: dict[str, int] | defaultdict[str, int]
        if cumulative:
            # self.cumulative_values[plot_id] is a defaultdict(int)
            # value is a dict[str, int]
            cumulative_dict_for_plot = self.cumulative_values[plot_id]
            for key, val_item in value.items():
                cumulative_dict_for_plot[key] += val_item
            value_to_log = cumulative_dict_for_plot
        else:
            value_to_log = value

        gloabal_step = self.global_step.get(plot_id, 0)
        self.writer.add_scalars(plot_id, value_to_log, gloabal_step)
        self.global_step[plot_id] += 1

    def update_figure(self, plot_id: str, figure: Figure):
        gloabal_step = self.global_step.get(plot_id, 0)
        self.writer.add_figure(plot_id, figure, gloabal_step, close=True)
        self.global_step[plot_id] += 1

    def close(self):
        self.writer.close()

# type: ignore
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import math

from Go.Go_uf import UnionFind
from Buffer import BufferElement
from gameserver_local_uf import GameServerGo

NUM_POSSIBLE_SCORES = int(30.5 * 2 + 1)


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
    def __init__(self, rows=2, cols=2):
        plt.ion()  # Enable interactive mode

        # Create a configurable subplot layout
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(12, 8))
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
    def __init__(self) -> None:
        self.possible_scores = np.linspace(-30.5, 30.5, NUM_POSSIBLE_SCORES)
        pass

    def heatmap(
        self,
        uf: UnionFind,
        uf_after: UnionFind | None,
        model_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        is_white: bool,
        server: GameServerGo,
        final_score: float,
        save_image: bool = False,
        save_path: str = "model_overlay.png",
    ):
        # If we're saving, switch to a non-interactive backend temporarily
        if save_image:
            current_backend = plt.get_backend()
            plt.switch_backend("Agg")

        pi, pi_opp, outcome_logits, ownership, score_logits = model_output

        value = outcome_logits[0][0].item()
        mu_score = outcome_logits[0][1].item()
        sigma_score = outcome_logits[0][2].item()
        score = server.go.get_score(uf, server.go.komi)

        score_diff = score["black"]["sum"] - score["white"]["sum"]
        score_normalized = score_diff * (-1 if is_white else 1)

        owner_normalized = ownership * 2 - 1

        fig, ax = plt.subplots(3, 2, figsize=(8, 8))
        fig.suptitle(
            f"Move Prediction for {'White' if is_white else 'Black'} with current score: {score_normalized} | "
            f"final score: {final_score}\n"
            f"Outcome Prediction: value: {value:.2f}, {r'$\mu$'}: {mu_score:.2f}, {r'$\sigma$'}: {sigma_score:.2f}"
        )

        # ownership prediction plot
        im = ax[0][0].imshow(
            owner_normalized.detach().cpu().squeeze().numpy(), cmap="PuOr", interpolation="nearest", vmin=-1, vmax=1
        )
        black_y, black_x = np.where(uf.state == 1)
        white_y, white_x = np.where(uf.state == 2)
        disabled_y, disabled_x = np.where(uf.state == 3)
        ax[0][0].scatter(black_x, black_y, c="black", s=200, alpha=1)
        ax[0][0].scatter(white_x, white_y, c="white", s=200, alpha=1)
        ax[0][0].scatter(disabled_x, disabled_y, c="#808080", s=200, alpha=1, marker="x")
        ax[0][0].set_title("Ownership prediction")
        fig.colorbar(im, ax=ax[0][0])

        # score cdf plot
        score_onehot = torch.zeros(NUM_POSSIBLE_SCORES)
        score_idx = int(NUM_POSSIBLE_SCORES / 2) + math.floor(score_normalized)
        score_onehot[score_idx] = 1.0

        target_cdf = torch.cumsum(score_onehot, dim=0)

        score_probs = F.softmax(score_logits.squeeze(), dim=0)
        predicted_cdf: torch.Tensor = torch.cumsum(score_probs, dim=0)

        ax[0][1].plot(predicted_cdf.detach().cpu().numpy(), label="Predicted CDF")
        ax[0][1].plot(target_cdf.numpy(), label="Target CDF")
        ax[0][1].set_title("score CDF prediction")

        # score pdf plot
        ax[1][1].bar(self.possible_scores, score_probs.detach().cpu().numpy(), label="Predicted PDF", alpha=0.5)
        ax[1][1].set_title("score PDF prediction")

        pi_np = pi.detach().cpu().squeeze().numpy()
        pi_opp_np = pi_opp.detach().cpu().squeeze().numpy()

        pi_board = pi_np[:-1].reshape((5, 5))
        pi_pass = pi_np[-1].item()
        pi_opp_board = pi_opp_np[:-1].reshape((5, 5))
        pi_opp_pass = pi_opp_np[-1]

        # own move policy plot
        move_im = ax[1][0].imshow(pi_board, cmap="RdYlGn", interpolation="nearest", vmin=0, vmax=1)
        ax[1][0].set_title(f"Own Move Policy, pass: {pi_pass:.2f}")
        fig.colorbar(move_im, ax=ax[1][0])

        # opponent move policy plot
        move_im = ax[2][0].imshow(pi_opp_board, cmap="RdYlGn", interpolation="nearest", vmin=0, vmax=1)
        ax[2][0].set_title(f"Opp Move Policy, pass: {pi_opp_pass:.2f}")
        fig.colorbar(move_im, ax=ax[2][0])

        if uf_after is not None:
            current_state = uf.state
            next_state = uf_after.state
            diff = current_state - next_state
            stone_y, stone_x = np.where(diff != 0)
            assert len(stone_x) == len(stone_y) == 1, "There should be only one stone placed"
            color = "black" if is_white else "white"
            ax[2][0].scatter(stone_x, stone_y, c=color, s=200, alpha=1)

        fig.tight_layout()

        if save_image:
            plt.savefig(f"out/{save_path}")
            plt.close(fig)  # Close the figure to prevent display
            plt.switch_backend(current_backend)  # Restore original backend

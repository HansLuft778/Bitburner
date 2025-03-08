import matplotlib.pyplot as plt
from collections import deque


class Plotter:

    def __init__(self):
        self.cumulative_reward_black = []
        self.cumulative_reward_white = []  # Add white player data
        self.loss = []
        self.value_loss = []
        self.policy_loss = []
        self.epsilon = []
        self.winrate = deque(maxlen=100)
        self.winrate_data = []

        plt.ion()  # Enable interactive mode
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
            2, 2, figsize=(12, 8)
        )

        # Reward subplot with both black and white lines
        (self.reward_line_black,) = self.ax1.plot(
            self.cumulative_reward_black, label="Black"
        )
        (self.reward_line_white,) = self.ax1.plot(
            self.cumulative_reward_white, label="White"
        )
        self.ax1.set_xlabel("Updates")
        self.ax1.set_ylabel("Cumulative Wins")
        self.ax1.set_title("Cumulative Wins Over Time")
        self.ax1.legend()

        # Loss subplot
        (self.loss_line,) = self.ax2.plot(self.loss)
        self.ax2.set_xlabel("Updates")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_title("Training Loss Over Time")

        # policy loss subplot
        (self.policy_loss_line,) = self.ax3.plot(self.policy_loss)
        self.ax3.set_xlabel("policy loss")
        self.ax3.set_ylabel("policy loss")
        self.ax3.set_title("policy loss Over Time")

        # value loss subplot
        (self.value_loss_line,) = self.ax4.plot(self.value_loss)
        self.ax4.set_xlabel("Value Loss")
        self.ax4.set_ylabel("Value Loss")
        self.ax4.set_title("Value Loss Over Time")

        plt.tight_layout()

    def downsample_data(self, data_list, max_size=2000):
        if len(data_list) > max_size:
            data_list = data_list[::2]
        return data_list

    def update_wins_black(self, new_reward: float):
        if self.cumulative_reward_black:
            self.cumulative_reward_black.append(
                new_reward + self.cumulative_reward_black[-1]
            )
        else:
            self.cumulative_reward_black.append(new_reward)

        self.cumulative_reward_black = self.downsample_data(
            self.cumulative_reward_black, max_size=200
        )
        self.reward_line_black.set_ydata(self.cumulative_reward_black)
        self.reward_line_black.set_xdata(range(len(self.cumulative_reward_black)))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_wins_white(self, new_reward: float):
        if self.cumulative_reward_white:
            self.cumulative_reward_white.append(
                new_reward + self.cumulative_reward_white[-1]
            )
        else:
            self.cumulative_reward_white.append(new_reward)

        self.cumulative_reward_white = self.downsample_data(
            self.cumulative_reward_white, max_size=200
        )

        self.reward_line_white.set_ydata(self.cumulative_reward_white)
        self.reward_line_white.set_xdata(range(len(self.cumulative_reward_white)))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_loss(self, new_loss: float):
        self.loss.append(float(new_loss))
        self.loss = self.downsample_data(self.loss)

        self.loss_line.set_ydata(self.loss)
        self.loss_line.set_xdata(range(len(self.loss)))
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_policy_loss(self, new_policy_loss: float):
        self.policy_loss.append(new_policy_loss)
        # self.epsilon = self.downsample_data(self.epsilon)

        self.policy_loss_line.set_ydata(self.policy_loss)
        self.policy_loss_line.set_xdata(range(len(self.policy_loss)))
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_value_loss(self, new_value_loss: float):
        self.value_loss.append(new_value_loss)
        # self.winrate_data.append(sum(self.winrate) / len(self.winrate))
        # self.winrate_data = self.downsample_data(self.winrate_data)

        self.value_loss_line.set_ydata(self.value_loss)
        self.value_loss_line.set_xdata(range(len(self.value_loss)))
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


"""
Average Q-Value (optional, more advanced)

You can record the agent's average or max Q-value for the chosen action each step. This can sometimes help diagnose if Q-values explode or collapse to some trivial value.
"""

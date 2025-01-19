import matplotlib.pyplot as plt


class Plotter:

    def __init__(self):
        self.cumulative_reward = []
        self.loss = []
        self.epsilon = []

        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))

        # Reward subplot
        (self.reward_line,) = self.ax1.plot(self.cumulative_reward)
        self.ax1.set_xlabel("Updates")
        self.ax1.set_ylabel("Cumulative Reward")
        self.ax1.set_title("Cumulative Reward Over Time")

        # Loss subplot
        (self.loss_line,) = self.ax2.plot(self.loss)
        self.ax2.set_xlabel("Updates")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_title("Training Loss Over Time")

        # epsilon subplot
        (self.epsilon_line,) = self.ax3.plot(self.epsilon)
        self.ax3.set_xlabel("Episodes")
        self.ax3.set_ylabel("Epsilon")
        self.ax3.set_title("Expsilon Over Time")

        plt.tight_layout()

    def update_reward(self, new_reward: float):
        if self.cumulative_reward:
            self.cumulative_reward.append(new_reward + self.cumulative_reward[-1])
        else:
            self.cumulative_reward.append(new_reward)
        self.reward_line.set_ydata(self.cumulative_reward)
        self.reward_line.set_xdata(range(len(self.cumulative_reward)))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_loss(self, new_loss: float):
        self.loss.append(float(new_loss))

        self.loss_line.set_ydata(self.loss)
        self.loss_line.set_xdata(range(len(self.loss)))
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_epsilon(self, new_eplsilon: float):
        self.epsilon.append(new_eplsilon)

        self.epsilon_line.set_ydata(self.epsilon)
        self.epsilon_line.set_xdata(range(len(self.epsilon)))
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

"""
Average Q-Value (optional, more advanced)

You can record the agent's average or max Q-value for the chosen action each step. This can sometimes help diagnose if Q-values explode or collapse to some trivial value.
"""
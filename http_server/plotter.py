import matplotlib.pyplot as plt
from collections import deque


class Plot:
    def __init__(self, ax, title, xlabel, ylabel, label=None, maxlen=None, color=None):
        self.data = [] if maxlen is None else deque(maxlen=maxlen)
        self.ax = ax
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.line, = self.ax.plot([], label=label, color=color)
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
    def __init__(self):
        plt.ion()  # Enable interactive mode
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
            2, 2, figsize=(12, 8)
        )
        
        # Dictionary to store all plots
        self.plots = {}
        
        # Initialize default plots
        self.add_plot("cumulative_reward_black", self.ax1, "Cumulative Wins Over Time", 
                     "Updates", "Cumulative Wins", label="Black")
        self.add_plot("cumulative_reward_white", self.ax1, "Cumulative Wins Over Time", 
                     "Updates", "Cumulative Wins", label="White")
        self.add_plot("loss", self.ax2, "Training Loss Over Time",
                     "Updates", "Loss")
        self.add_plot("policy_loss", self.ax3, "Policy Loss Over Time", 
                     "Updates", "Policy Loss")
        self.add_plot("value_loss", self.ax4, "Value Loss Over Time",
                     "Updates", "Value Loss")
        
        # Legacy attributes to maintain compatibility
        self.cumulative_reward_black = self.plots["cumulative_reward_black"].data
        self.cumulative_reward_white = self.plots["cumulative_reward_white"].data
        self.loss = self.plots["loss"].data
        self.policy_loss = self.plots["policy_loss"].data
        self.value_loss = self.plots["value_loss"].data
        
        # Initialize winrate tracking
        self.winrate = deque(maxlen=100)
        self.winrate_data = []
        
        plt.tight_layout()
        
    def add_plot(self, name, ax, title, xlabel, ylabel, label=None, maxlen=None, color=None):
        """Add a new plot to be tracked"""
        self.plots[name] = Plot(ax, title, xlabel, ylabel, label, maxlen, color)
        return self.plots[name]
        
    def update_stat(self, name, value, cumulative=False, max_size=2000):
        """Generic method to update any stat"""
        if name not in self.plots:
            raise ValueError(f"Plot '{name}' not found. Add it first with add_plot().")
            
        self.plots[name].update(value, cumulative, max_size)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    # Legacy methods for backward compatibility
    def update_wins_black(self, new_reward: float):
        self.update_stat("cumulative_reward_black", new_reward, cumulative=True, max_size=200)
        
    def update_wins_white(self, new_reward: float):
        self.update_stat("cumulative_reward_white", new_reward, cumulative=True, max_size=200)
        
    def update_loss(self, new_loss: float):
        self.update_stat("loss", new_loss)
        
    def update_policy_loss(self, new_policy_loss: float):
        self.update_stat("policy_loss", new_policy_loss)
        
    def update_value_loss(self, new_value_loss: float):
        self.update_stat("value_loss", new_value_loss)
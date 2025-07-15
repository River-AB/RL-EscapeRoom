import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlotViewer:
    """
    A class to manage a matplotlib figure with multiple plots that can be cycled through.
    This ensures plots are drawn one at a time and do not overlap.
    """
    def __init__(self, agent_name, grid, plot_data):
        self.agent_name = agent_name
        self.grid = grid
        self.plot_data = plot_data
        self.current_plot_index = 0
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig = plt.figure(figsize=(12, 12)) # Adjusted figure size for a square grid
        
        self._add_buttons()
        self.draw_current_plot()
        plt.show()

    def _add_buttons(self):
        # Position buttons at the bottom of the figure
        ax_prev = self.fig.add_axes([0.7, 0.01, 0.1, 0.05])
        ax_next = self.fig.add_axes([0.81, 0.01, 0.1, 0.05])
        from matplotlib.widgets import Button # Import here to keep it local
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_plot)
        self.btn_next.on_clicked(self.next_plot)

    def draw_current_plot(self):
        # Clear the entire figure to prevent overlaps
        self.fig.clear()
        
        plot_info = self.plot_data[self.current_plot_index]
        plot_type = plot_info['type']
        data = plot_info['data']
        
        # Use a GridSpec for better layout control
        gs = self.fig.add_gridspec(1, 1, bottom=0.1)
        self.ax = self.fig.add_subplot(gs[0, 0])
        
        if plot_type == 'rewards':
            self._plot_rewards(data)
        elif plot_type == 'steps':
            self._plot_steps(data)
        elif plot_type == 'action_schema':
            self._plot_action_schema(data)

        # Re-add buttons after clearing
        self._add_buttons()
        self.fig.canvas.draw_idle()

    def _plot_rewards(self, rewards_data):
        self.ax.set_title(f'Episodic Rewards for {self.agent_name}', fontsize=18, weight='bold')
        rewards_series = pd.Series(rewards_data)
        moving_avg_rewards = rewards_series.rolling(window=100, min_periods=1).mean()
        
        self.ax.plot(rewards_series, color='lightblue', alpha=0.7, label='Total Reward per Episode')
        self.ax.plot(moving_avg_rewards, color='crimson', linewidth=2.5, label='100-Episode Moving Average')
        self.ax.set_ylabel("Total Reward", fontsize=14)
        self.ax.set_xlabel("Episode", fontsize=14)
        self.ax.legend(fontsize=12)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    def _plot_steps(self, steps_data):
        self.ax.set_title(f'Episodic Steps for {self.agent_name}', fontsize=18, weight='bold')
        steps_series = pd.Series(steps_data)
        moving_avg_steps = steps_series.rolling(window=100, min_periods=1).mean()

        self.ax.plot(steps_series, color='mediumseagreen', alpha=0.7, label='Steps per Episode')
        self.ax.plot(moving_avg_steps, color='darkgreen', linewidth=2.5, label='100-Episode Moving Average')
        self.ax.set_xlabel("Episode", fontsize=14)
        self.ax.set_ylabel("Steps", fontsize=14)
        self.ax.legend(fontsize=12)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    def _plot_action_schema(self, action_counts):
        self.ax.set_title(f'Action Frequencies per Cell', fontsize=18, weight='bold', pad=20)
        size = self.grid.shape[0]
        
        # Set up the grid appearance
        self.ax.set_xlim(-0.5, size - 0.5)
        self.ax.set_ylim(-0.5, size - 0.5)
        self.ax.set_xticks(np.arange(size))
        self.ax.set_yticks(np.arange(size))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        
        # Draw grid lines
        for i in range(size + 1):
            self.ax.axhline(i - 0.5, color='k', linewidth=1.5)
            self.ax.axvline(i - 0.5, color='k', linewidth=1.5)

        colors = {'up': 'red', 'down': 'blue', 'left': 'purple', 'right': 'black'}
        
        # Define precise positions within the cell for each action's text
        positions = {
            'up': {'x': 0, 'y': -0.25, 'va': 'center'},
            'down': {'x': 0, 'y': 0.25, 'va': 'center'},
            'left': {'x': -0.25, 'y': 0, 'ha': 'center'},
            'right': {'x': 0.25, 'y': 0, 'ha': 'center'}
        }

        for r in range(size):
            for c in range(size):
                cell_counts = action_counts.get((r, c), {})
                
                # Draw the counts for each action in its designated position
                for action, pos in positions.items():
                    if action in cell_counts:
                        count = cell_counts[action]
                        self.ax.text(
                            c + pos['x'], 
                            r + pos['y'], 
                            str(count),
                            ha=pos.get('ha', 'center'),
                            va=pos.get('va', 'center'),
                            color=colors[action],
                            fontsize=9,
                            weight='bold'
                        )
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    def next_plot(self, event):
        self.current_plot_index = (self.current_plot_index + 1) % len(self.plot_data)
        self.draw_current_plot()

    def prev_plot(self, event):
        self.current_plot_index = (self.current_plot_index - 1) % len(self.plot_data)
        self.draw_current_plot()

def show_plots(agent_name, grid, plot_data):
    """Entry point for the plotting process."""
    PlotViewer(agent_name, grid, plot_data)

import numpy as np
import random
from constants import WALL, EXIT, EMPTY, START

class BaseAgent:
    """A base class that defines the interface for all agents."""
    def __init__(self, env, settings):
        self.env = env
        self.is_trained = False
        self.name = "Base Agent"
        self.training_type = "iterative" 
        self.episode_rewards = []
        self.episode_steps = []
        
    @staticmethod
    def get_editor_options():
        return {}
        
    def reset(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def extract_policy(self):
        raise NotImplementedError

class BaseRoom:
    """A base class that defines the interface for all rooms."""
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_pos = (0, 0)
        self.exit_pos = (size - 1, size - 1)
        self.state_space = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.action_space = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.name = "Base Room"
        self.slippery_probabilities = {}

    def generate_layout(self, settings):
        self.grid.fill(EMPTY)
        self.grid[self.start_pos] = START
        self.grid[self.exit_pos] = EXIT
        
        possible_placements = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in [self.start_pos, self.exit_pos]]
        
        num_walls_str = settings.get('Walls', '0')
        num_walls = random.randint(1,15) if "Random" in num_walls_str else int(num_walls_str)
        num_walls = min(num_walls, len(possible_placements))
        wall_positions = random.sample(possible_placements, num_walls)
        for pos in wall_positions:
            self.grid[pos] = WALL
            if pos in possible_placements:
                possible_placements.remove(pos)

    def get_valid_actions(self, state):
        """Returns a list of valid actions from a given state."""
        actions = []
        r, c = state[0], state[1] # Unpack player position from state
        for action, (dr, dc) in self.action_space.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != WALL:
                actions.append(action)
        return actions

    def reset_state(self):
        pass

    def get_state_type(self, state_pos):
        return self.grid[state_pos]

    def get_transition_model(self, state, action):
        raise NotImplementedError

    def get_editor_options(self):
        return {}
        
    def step(self, state, action):
        """A simple step function for model-free agents."""
        state_pos = (state[0], state[1])
        if self.get_state_type(state_pos) in [WALL, EXIT]:
            return state, 0, True

        d_row, d_col = self.action_space[action]
        next_row, next_col = state[0] + d_row, state[1] + d_col

        if not (0 <= next_row < self.size and 0 <= next_col < self.size and self.grid[next_row, next_col] != WALL):
            return state, -10.0, False 

        next_state = (next_row, next_col)
        
        if next_state == self.exit_pos:
            return next_state, 100.0, True
        
        return next_state, 0.0, False

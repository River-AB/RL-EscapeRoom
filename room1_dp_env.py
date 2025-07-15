import numpy as np
import random
from base_classes import BaseRoom
from constants import EMPTY, WALL, START, EXIT, SLIPPERY, BAG, ROPE

class FirstEscapeRoom(BaseRoom):
    """Room 1: Walls, slippery tiles, and items to collect in order."""
    def __init__(self, size=10):
        super().__init__(size)
        self.name = "Room 1: Dynamic Programming"
        self.bag_pos = None
        self.rope_pos = None
        self.state_space = [(r, c, b, r_p) for r in range(size) for c in range(size) for b in range(2) for r_p in range(2)]
        self.slippery_probabilities = {}

    def get_editor_options(self):
        return {
            "Walls": {"type": "dropdown", "options": ["Random"] + list(range(21)), "default": "Random"},
            "Slippery Tiles": {"type": "dropdown", "options": ["Random"] + list(range(21)), "default": "Random"},
            "Start with Items": {"type": "dropdown", "options": ["None", "Bag", "Rope", "Both"], "default": "None"},
        }

    def generate_layout(self, settings):
        """Generates the layout with items spawning in a central 4x4 area."""
        super().generate_layout(settings)
        self.slippery_probabilities = {} # Reset for new map
        
        zone_size = 4
        start_coord = (self.size - zone_size) // 2
        end_coord = start_coord + zone_size
        
        item_spawn_zone = [(r, c) for r in range(start_coord, end_coord) for c in range(start_coord, end_coord)]

        possible_placements = [p for p in item_spawn_zone if self.grid[p] == EMPTY]

        if len(possible_placements) >= 2:
            pos1, pos2 = random.sample(possible_placements, 2)
            self.bag_pos = pos1
            self.rope_pos = pos2
        else:
            self.bag_pos, self.rope_pos = None, None

        if self.bag_pos: self.grid[self.bag_pos] = BAG
        if self.rope_pos: self.grid[self.rope_pos] = ROPE

        possible_placements_2d = [(r,c) for r,c in np.ndindex(self.grid.shape) if self.grid[r,c] == EMPTY]
        
        num_slippery_str = settings.get('Slippery Tiles', 'Random')
        num_slippery = random.randint(1, 10) if num_slippery_str == "Random" else int(num_slippery_str)

        num_slippery = min(num_slippery, len(possible_placements_2d))
        slippery_positions = random.sample(possible_placements_2d, num_slippery)
        for pos in slippery_positions:
            self.grid[pos] = SLIPPERY
            self._generate_slippery_probabilities(pos)

        if not self._is_path_possible():
            self.generate_layout(settings)

    def _generate_slippery_probabilities(self, pos):
        """
        Generates random probabilities for slipping in each valid direction.
        """
        valid_actions = []
        for action, (dr, dc) in self.action_space.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != WALL:
                valid_actions.append(action)
        
        num_valid = len(valid_actions)
        if num_valid > 0:
            # Assign random weights to each valid direction
            weights = [0.2 + random.random() for _ in range(num_valid)]
            total_weight = sum(weights)
            
            if total_weight > 0:
                final_probs = {action: weights[i] / total_weight for i, action in enumerate(valid_actions)}
            else: # Should be rare, but handle case where all random numbers are 0
                final_probs = {action: 1.0 / num_valid for action in valid_actions}
        else:
            final_probs = {}
    
        self.slippery_probabilities[pos] = {action: final_probs.get(action, 0.0) for action in self.action_space}

    def get_transition_model(self, state, action):
        r, c, has_bag, has_rope = state
        
        if self.grid[r, c] in [WALL, EXIT]: return [(1.0, state, 0)]
        current_pos = (r, c)
        
        if self.grid[current_pos] == SLIPPERY:
            # On a slippery tile, the outcome is determined by the tile's random probabilities,
            # not the agent's intended action.
            tile_probs = self.slippery_probabilities.get(current_pos, {})
            outcomes = [(prob, act) for act, prob in tile_probs.items() if prob > 0]
        else:
            # On a normal tile, the outcome is deterministic based on the chosen action.
            outcomes = [(1.0, action)]

        transitions = []
        for prob, act in outcomes:
            if act is None:
                next_r, next_c = r, c
            else:
                d_row, d_col = self.action_space[act]
                next_r, next_c = r + d_row, c + d_col

            if 0 <= next_r < self.size and 0 <= next_c < self.size and self.grid[next_r, next_c] != WALL:
                next_state_pos = (next_r, next_c)
                reward = 0.0
            else:
                next_state_pos = (r, c)
                reward = -10.0
            
            next_has_bag, next_has_rope = has_bag, has_rope
            if next_state_pos == self.bag_pos and not has_bag:
                reward += 20; next_has_bag = 1
            elif next_state_pos == self.rope_pos and not has_rope:
                reward += 30 if has_bag else -10; next_has_rope = 1
            
            if next_state_pos == self.exit_pos:
                reward += 100 if next_has_bag and next_has_rope else -20

            final_next_state = (next_state_pos[0], next_state_pos[1], next_has_bag, next_has_rope)
            transitions.append((prob, final_next_state, reward))
        return transitions

    def step(self, state, action):
        transitions = self.get_transition_model(state, action)
        probabilities = [t[0] for t in transitions]
        next_states = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]
        
        chosen_index = random.choices(range(len(next_states)), weights=probabilities, k=1)[0]
        
        next_state = next_states[chosen_index]
        reward = rewards[chosen_index]
        done = (next_state[0], next_state[1]) == self.exit_pos
        return next_state, reward, done

    def _is_path_possible(self):
        q = [self.start_pos]
        visited = {self.start_pos}
        while q:
            r, c = q.pop(0)
            if (r, c) == self.exit_pos: return True
            for dr, dc in self.action_space.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != WALL and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

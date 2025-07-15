import numpy as np
import random
from base_classes import BaseRoom
from constants import EMPTY, WALL, START, EXIT, IRON_KEY, PORTAL, SLIPPERY

class SecondEscapeRoom(BaseRoom):
    """
    Room 2: A dynamic room with a patrolling enemy.
    """
    def __init__(self, size=10):
        super().__init__(size)
        self.name = "Room 2: SARSA with Enemy"
        self.state_space = [(pr, pc, hk) for pr in range(size) for pc in range(size) for hk in range(2)]
        self.key_pos = None
        self.door_pos = None
        self.portal_in_pos = None
        self.portal_out_pos = None
        self.has_key = False
        
        self.enemy_pos = (0, 0)
        self.patrol_route = []
        self.patrol_index = 0

        self.original_grid = np.zeros((size, size), dtype=int)
        self.slippery_probabilities = {}

    def get_editor_options(self):
        return {
            "Walls": {"type": "dropdown", "options": ["Random"] + list(range(21)), "default": "Random"},
            "Slippery Tiles": {"type": "dropdown", "options": ["Random"] + list(range(21)), "default": "Random"},
        }

    def generate_layout(self, settings):
        """
        Generates the layout for Room 2.
        """
        self.slippery_probabilities = {}
        self.grid = np.zeros((self.size, self.size), dtype=int)

        self.grid[4, :] = WALL
        self.grid[4:, 6] = WALL

        self._generate_patrol_route()
        
        occupied_positions = {tuple(p) for p in np.argwhere(self.grid != EMPTY)}
        occupied_positions.update(self.patrol_route)
        occupied_positions.add(self.start_pos)
        occupied_positions.add(self.exit_pos)

        key_zone = [(r, c) for r in range(4) for c in range(self.size)]
        possible_key_placements = [p for p in key_zone if p not in occupied_positions]
        if possible_key_placements:
            self.key_pos = random.choice(possible_key_placements)
        else:
            fallback_options = [p for p in np.ndindex(self.size, self.size) if p not in occupied_positions]
            self.key_pos = random.choice(fallback_options) if fallback_options else (0, 1)
        if self.key_pos: 
            occupied_positions.add(self.key_pos)

        door_candidates = [(4, c) for c in range(1, 6) if self.grid[4, c] == WALL]
        self.door_pos = random.choice(door_candidates) if door_candidates else (4, 1)

        portal_in_zone = [(r, c) for r in range(7, 10) for c in range(3)]
        portal_out_zone = [(r, c) for r in range(5, 7) for c in range(7, 10)]
        possible_portal_in = [p for p in portal_in_zone if p not in occupied_positions]
        self.portal_in_pos = random.choice(possible_portal_in) if possible_portal_in else None
        if self.portal_in_pos: occupied_positions.add(self.portal_in_pos)
        possible_portal_out = [p for p in portal_out_zone if p not in occupied_positions]
        self.portal_out_pos = random.choice(possible_portal_out) if possible_portal_out else None
        if self.portal_out_pos: occupied_positions.add(self.portal_out_pos)

        self.grid[self.start_pos] = START
        self.grid[self.exit_pos] = EXIT
        if self.key_pos: self.grid[self.key_pos] = IRON_KEY
        if self.door_pos: self.grid[self.door_pos] = WALL
        if self.portal_in_pos: self.grid[self.portal_in_pos] = PORTAL
        if self.portal_out_pos: self.grid[self.portal_out_pos] = PORTAL
        
        num_walls_str = settings.get('Walls', 'Random')
        num_slippery_str = settings.get('Slippery Tiles', 'Random')

        num_walls = random.randint(1, 10) if num_walls_str == "Random" else int(num_walls_str)
        num_slippery = random.randint(1, 10) if num_slippery_str == "Random" else int(num_slippery_str)

        possible_randoms = [(r, c) for r,c in np.ndindex(self.grid.shape) if self.grid[r,c] == EMPTY and (r,c) not in occupied_positions]
        
        wall_positions = random.sample(possible_randoms, min(num_walls, len(possible_randoms)))
        for pos in wall_positions: self.grid[pos] = WALL
        possible_randoms = [p for p in possible_randoms if p not in wall_positions]

        slippery_positions = random.sample(possible_randoms, min(num_slippery, len(possible_randoms)))
        for pos in slippery_positions:
            self.grid[pos] = SLIPPERY

        all_slippery_tiles = np.argwhere(self.grid == SLIPPERY)
        for pos_array in all_slippery_tiles:
            pos_tuple = tuple(pos_array)
            self._generate_slippery_probabilities(pos_tuple)

        self.original_grid = np.copy(self.grid)
        self.reset_state()

    def _generate_patrol_route(self):
        path = []
        for c in range(self.size - 1, -1, -1): path.append((0, c))
        for r in range(1, 4): path.append((r, 0))
        for c in range(1, self.size): path.append((3, c))
        for r in range(2, -1, -1): path.append((r, self.size - 1))
        self.patrol_route = path
        
    def reset_state(self):
        self.grid = np.copy(self.original_grid)
        self.has_key = False
        self.patrol_index = 0
        self.enemy_pos = self.patrol_route[0] if self.patrol_route else (0, self.size-1)
        if self.key_pos: self.grid[self.key_pos] = IRON_KEY
        if self.door_pos: self.grid[self.door_pos] = WALL

    def step(self, state, action):
        player_r, player_c, has_key_state = state
        
        current_pos = (player_r, player_c)
        next_player_pos = current_pos
        reward = 0.0 
        
        if self.grid[current_pos] == SLIPPERY:
            probs = self.slippery_probabilities.get(current_pos, {})
            if probs:
                valid_slip_actions = [act for act, p in probs.items() if p > 0]
                if valid_slip_actions:
                    action = random.choice(valid_slip_actions)

        d_row, d_col = self.action_space[action]
        nr, nc = player_r + d_row, player_c + d_col
        
        if not (0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != WALL):
            reward = -5.0 
            next_player_pos = current_pos
        else:
            next_player_pos = (nr, nc)
        
        if next_player_pos == self.enemy_pos:
            return (*next_player_pos, has_key_state), -100.0, True

        if self.patrol_route:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_route)
            self.enemy_pos = self.patrol_route[self.patrol_index]

        if next_player_pos == self.enemy_pos:
            return (*next_player_pos, has_key_state), -100.0, True

        if next_player_pos == self.portal_in_pos:
            next_player_pos = self.portal_out_pos
            reward += 5.0
        
        tile_at_next_pos = self.grid[next_player_pos]
        next_has_key_state = has_key_state
        if tile_at_next_pos == IRON_KEY and not has_key_state:
            self.has_key = True; next_has_key_state = 1; reward += 50.0 
            if self.key_pos: self.grid[self.key_pos] = EMPTY
            if self.door_pos: self.grid[self.door_pos] = EMPTY
        
        done = (next_player_pos == self.exit_pos)
        if done: reward += 100.0
            
        next_state = (*next_player_pos, next_has_key_state)
        return next_state, reward, done

    def _generate_slippery_probabilities(self, pos):
        """
        Calculates and stores the probability of slipping in each valid direction.
        """
        valid_actions = self.get_valid_actions(pos)
        num_valid = len(valid_actions)
        
        final_probs = {}

        if num_valid > 0:
            prob = 1.0 / num_valid
            final_probs = {action: prob for action in valid_actions}
        
        self.slippery_probabilities[pos] = {action: final_probs.get(action, 0.0) for action in self.action_space}

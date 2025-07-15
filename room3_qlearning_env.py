import numpy as np
import random
from base_classes import BaseRoom
from constants import EMPTY, WALL, START, EXIT, PLANK, POTHOLE, BRIDGE, LOCKED_DOOR, SILVER_KEY, GOLDEN_KEY

class ThirdEscapeRoom(BaseRoom):
    """
    Room 3 Challenge: The Pothole Islands.
    """
    def __init__(self, size=10):
        super().__init__(size)
        self.name = "Room 3: The Pothole Islands"
        
        self.action_space = {
            'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1),
            'pull_up': (-1, 0), 'pull_down': (1, 0), 'pull_left': (0, -1), 'pull_right': (0, 1)
        }
        self.state_space = "Too large to compute, defined by interactions."

        self.original_plank1_pos = None
        self.original_plank2_pos = None
        self.silver_key_pos = None
        self.golden_key_pos = None
        self.locked_door_pos = None
        self.original_grid = np.zeros((size, size), dtype=int)
        self.silver_key_access_points = set()
        self.golden_key_access_points = set()


    def get_editor_options(self):
        """Returns editor options for walls."""
        return {
            "Walls": {"type": "dropdown", "options": ["Random"] + list(range(21)), "default": "0"},
        }

    def generate_layout(self, settings):
        """Generates a randomized layout with two pothole islands, two planks, and a gated exit."""
        self.grid.fill(EMPTY)
        occupied_coords = set()

        self.start_pos = (0, 0)
        occupied_coords.add(self.start_pos)
        
        self.exit_pos = (self.size - 1, self.size - 1)
        occupied_coords.add(self.exit_pos)

        self.grid[self.size - 2, self.size - 2] = WALL
        occupied_coords.add((self.size - 2, self.size - 2))

        if random.random() < 0.5:
            self.locked_door_pos = (self.size - 2, self.size - 1)
            self.grid[self.size - 1, self.size - 2] = WALL
            occupied_coords.add((self.size - 1, self.size - 2))
            door_access_point = (self.locked_door_pos[0] - 1, self.locked_door_pos[1])
        else:
            self.locked_door_pos = (self.size - 1, self.size - 2)
            self.grid[self.size - 2, self.size - 1] = WALL
            occupied_coords.add((self.size - 2, self.size - 1))
            door_access_point = (self.locked_door_pos[0], self.locked_door_pos[1] - 1)
        
        occupied_coords.add(self.locked_door_pos)
        if 0 <= door_access_point[0] < self.size and 0 <= door_access_point[1] < self.size:
             occupied_coords.add(door_access_point)

        island_spawn_zone = [(r, c) for r in range(2, self.size - 4) for c in range(2, self.size - 4)]
        random.shuffle(island_spawn_zone)
        
        island1_tl = None
        for potential_tl in island_spawn_zone:
            potential_coords = {(r, c) for r in range(potential_tl[0], potential_tl[0] + 3) for c in range(potential_tl[1], potential_tl[1] + 3)}
            if not potential_coords.intersection(occupied_coords):
                island1_tl = potential_tl
                occupied_coords.update(potential_coords)
                break
        
        if island1_tl is None: self.generate_layout(settings); return

        island2_tl = None
        for potential_tl in island_spawn_zone:
            potential_coords = {(r, c) for r in range(potential_tl[0], potential_tl[0] + 3) for c in range(potential_tl[1], potential_tl[1] + 3)}
            if not potential_coords.intersection(occupied_coords):
                island2_tl = potential_tl
                occupied_coords.update(potential_coords)
                break
        
        if island2_tl is None: self.generate_layout(settings); return
            
        island1_coords = {(r, c) for r in range(island1_tl[0], island1_tl[0] + 3) for c in range(island1_tl[1], island1_tl[1] + 3)}
        island2_coords = {(r, c) for r in range(island2_tl[0], island2_tl[0] + 3) for c in range(island2_tl[1], island2_tl[1] + 3)}
        
        for r, c in island1_coords: self.grid[r, c] = POTHOLE
        for r, c in island2_coords: self.grid[r, c] = POTHOLE
        
        self.silver_key_pos = (island1_tl[0] + 1, island1_tl[1] + 1)
        self.golden_key_pos = (island2_tl[0] + 1, island2_tl[1] + 1)
        self.grid[self.silver_key_pos] = SILVER_KEY
        self.grid[self.golden_key_pos] = GOLDEN_KEY

        skr, skc = self.silver_key_pos
        self.silver_key_access_points = {(skr - 1, skc), (skr + 1, skc), (skr, skc - 1), (skr, skc + 1)}
        gkr, gkc = self.golden_key_pos
        self.golden_key_access_points = {(gkr - 1, gkc), (gkr + 1, gkc), (gkr, gkc - 1), (gkr, gkc + 1)}

        plank_spawn_zone = [(r, c) for r in range(self.size) for c in range(self.size)]
        valid_plank_placements = [pos for pos in plank_spawn_zone if pos not in occupied_coords]
        
        if len(valid_plank_placements) < 2: self.generate_layout(settings); return
            
        self.original_plank1_pos, self.original_plank2_pos = random.sample(valid_plank_placements, 2)
        occupied_coords.add(self.original_plank1_pos)
        occupied_coords.add(self.original_plank2_pos)

        possible_randoms = [p for p in np.ndindex(self.grid.shape) if p not in occupied_coords]
        
        num_walls_str = settings.get('Walls', '0')
        num_walls = random.randint(1, 10) if num_walls_str == "Random" else int(num_walls_str)
        wall_positions = random.sample(possible_randoms, min(num_walls, len(possible_randoms)))
        for pos in wall_positions:
            self.grid[pos] = WALL

        self.grid[self.start_pos] = START
        self.grid[self.exit_pos] = EXIT
        self.grid[self.locked_door_pos] = LOCKED_DOOR
        
        self.original_grid = np.copy(self.grid)
        self.reset_state()
    
    def encode_bridge_pos(self, r, c):
        return r * self.size + c

    def decode_bridge_pos(self, encoded_pos):
        r = encoded_pos // self.size
        c = encoded_pos % self.size
        return (r, c)

    def reset_state(self):
        self.grid = np.copy(self.original_grid)

    def get_valid_actions(self, state):
        player_r, player_c, p1_r, p1_c, p2_r, p2_c, has_silver, has_golden = state
        
        temp_grid = np.copy(self.grid)
        
        if p1_r == -1: temp_grid[self.decode_bridge_pos(p1_c)] = BRIDGE
        if p2_r == -1: temp_grid[self.decode_bridge_pos(p2_c)] = BRIDGE

        actions = []
        for action, (dr, dc) in self.action_space.items():
            nr, nc = player_r + dr, player_c + dc
            
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue

            tile_type = temp_grid[nr, nc]
            is_wall = tile_type == WALL
            is_locked_door = tile_type == LOCKED_DOOR and not (has_silver and has_golden)

            if is_wall or is_locked_door:
                continue
                
            actions.append(action)
        return actions

    def step(self, state, action):
        player_r, player_c, p1_r, p1_c, p2_r, p2_c, has_silver, has_golden = state
        reward = -0.1
        
        temp_grid = np.copy(self.grid)
        if p1_r == -1: temp_grid[self.decode_bridge_pos(p1_c)] = BRIDGE
        if p2_r == -1: temp_grid[self.decode_bridge_pos(p2_c)] = BRIDGE

        d_row, d_col = self.action_space[action]
        
        if "pull" in action:
            plank_to_pull_pos = (player_r + d_row, player_c + d_col)
            agent_new_pos = (player_r - d_row, player_c - d_col)

            if not (0 <= agent_new_pos[0] < self.size and 0 <= agent_new_pos[1] < self.size):
                return state, -5.0, False 

            tile_to_step_on = temp_grid[agent_new_pos]
            is_locked_door = agent_new_pos == self.locked_door_pos and not (has_silver and has_golden)

            if tile_to_step_on == WALL or is_locked_door:
                return state, -5.0, False
            if tile_to_step_on == POTHOLE:
                return state, -100.0, True

            planks = [([p1_r, p1_c], 1), ([p2_r, p2_c], 2)]
            for i, (plank_state, plank_id) in enumerate(planks):
                if plank_state[0] != -1 and tuple(plank_state) == plank_to_pull_pos:
                    new_plank_pos = (player_r, player_c)
                    other_plank_state = planks[1-i][0]
                    
                    s_p1, s_p2 = (new_plank_pos, other_plank_state) if plank_id == 1 else (other_plank_state, new_plank_pos)
                    next_state = (*agent_new_pos, *s_p1, *s_p2, has_silver, has_golden)
                    return next_state, 0.0, False
            return state, -5.0, False

        next_player_r, next_player_c = player_r + d_row, player_c + d_col

        planks = [([p1_r, p1_c], 1), ([p2_r, p2_c], 2)]
        for i, (plank_state, plank_id) in enumerate(planks):
             if plank_state[0] != -1 and (next_player_r, next_player_c) == tuple(plank_state):
                next_plank_r, next_plank_c = plank_state[0] + d_row, plank_state[1] + d_col
                
                if not (0 <= next_plank_r < self.size and 0 <= next_plank_c < self.size): return state, -5.0, False
                tile_beyond_plank = temp_grid[next_plank_r, next_plank_c]
                
                if tile_beyond_plank == POTHOLE:
                    new_bridge_pos = (next_plank_r, next_plank_c)
                    is_new_bridge_for_silver = new_bridge_pos in self.silver_key_access_points
                    is_new_bridge_for_golden = new_bridge_pos in self.golden_key_access_points
                    if not is_new_bridge_for_silver and not is_new_bridge_for_golden: return state, -20.0, False

                    other_plank_state = planks[1-i][0]
                    if other_plank_state[0] == -1:
                        existing_bridge_pos = self.decode_bridge_pos(other_plank_state[1])
                        is_existing_bridge_for_silver = existing_bridge_pos in self.silver_key_access_points
                        is_existing_bridge_for_golden = existing_bridge_pos in self.golden_key_access_points
                        if (is_new_bridge_for_silver and is_existing_bridge_for_silver) or (is_new_bridge_for_golden and is_existing_bridge_for_golden):
                           return state, -50.0, False
                    reward += 30.0
                    encoded_pos = self.encode_bridge_pos(next_plank_r, next_plank_c)
                    new_plank_state = [-1, encoded_pos]
                    s_p1, s_p2 = (new_plank_state, other_plank_state) if plank_id == 1 else (other_plank_state, new_plank_state)
                    next_state = (next_player_r, next_player_c, *s_p1, *s_p2, has_silver, has_golden)
                    return next_state, reward, False
                elif tile_beyond_plank == EMPTY:
                    new_plank_state = [next_plank_r, next_plank_c]
                    other_plank_state = planks[1-i][0]
                    s_p1, s_p2 = (new_plank_state, other_plank_state) if plank_id == 1 else (other_plank_state, new_plank_state)
                    next_state = (next_player_r, next_player_c, *s_p1, *s_p2, has_silver, has_golden)
                    return next_state, 0.0, False
                else: return state, -5.0, False

        if not (0 <= next_player_r < self.size and 0 <= next_player_c < self.size): return state, -5.0, False
        tile_type = temp_grid[next_player_r, next_player_c]
        if tile_type == WALL or (tile_type == LOCKED_DOOR and not (has_silver and has_golden)): return state, -5.0, False
        if tile_type == POTHOLE: return state, -100.0, True

        if (next_player_r, next_player_c) == self.silver_key_pos and not has_silver: has_silver = 1; reward += 50.0
        if (next_player_r, next_player_c) == self.golden_key_pos and not has_golden: has_golden = 1; reward += 50.0

        next_state = (next_player_r, next_player_c, p1_r, p1_c, p2_r, p2_c, has_silver, has_golden)
        if (next_player_r, next_player_c) == self.exit_pos: return next_state, 100.0, True
            
        return next_state, reward, False

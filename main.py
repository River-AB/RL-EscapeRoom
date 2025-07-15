import pygame
import time
import numpy as np
import random
import os
import multiprocessing
from collections import deque, defaultdict

from ui_components import UIManager
from sprite_handler import AnimatedSprite
from editor_menu import EditorMenu
from plot_utils import show_plots
from constants import *

# Import all rooms and agents
from room1_dp_env import FirstEscapeRoom
from room2_sarsa_env import SecondEscapeRoom
from room3_qlearning_env import ThirdEscapeRoom
from agent_dp import DynamicProgrammingAgent
from agent_sarsa import SarsaAgent
from agent_qlearning import QLearningAgent

class PygameVisualizer:
    def __init__(self):
        pygame.init()
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Reinforcement Learning Playground")
        self.clock = pygame.time.Clock()
        self.item_font = pygame.font.SysFont('Arial', 30, bold=True)
        self.prob_font = pygame.font.SysFont('Monospace', 11, bold=True)
        self.q_value_font = pygame.font.SysFont('Monospace', 10, bold=True)
        self.console_font = pygame.font.SysFont('Monospace', 14)
        self.button_font = pygame.font.SysFont('Arial', 18, bold=True)
        self.status_font = pygame.font.SysFont('Arial', 22, bold=True)
        self.editor_font = pygame.font.SysFont('Arial', 18)
        self.popup_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        self.running = True
        self.is_animating = False
        self.is_paused = False
        self.is_slow_training = False
        self.is_training_paused = False
        self.success_popup_active = False
        self.show_q_values = False
        self.show_skip_episode_input = False
        self.skip_episode_input_text = ""
        self.death_animation_sequence = None
        
        self.console_logs = []
        self.console_scroll_offset_y = 0

        self.animation_timer = 0
        self.animation_delay = 350
        self.training_timer = 0
        self.training_delay = 500 # Default value
        
        self.training_iteration = 0
        self.editor_settings = {}
        self.prev_enemy_pos = (0,0)
        self.using_image_assets = True

        self.console_rect = pygame.Rect(GRID_WIDTH, 0, CONSOLE_WIDTH, GRID_WIDTH)
        self.log_message("Welcome to the RL Playground!")
        self.log_message("Press F11 to toggle fullscreen.")
        
        self.hero_sprite = AnimatedSprite(size=CELL_SIZE, sprite_name='Hero')
        self.enemy_sprite = AnimatedSprite(size=CELL_SIZE, sprite_name='Enemy')
        self.ui_manager = UIManager(self.screen, self.button_font, COLORS)
        self._load_item_images()

        self._setup_rooms_and_agents()
        self.load_room(0)

    def _create_fallback_surface(self, color, text=""):
        """Creates a fallback surface with a color and optional text."""
        surface = pygame.Surface((CELL_SIZE, CELL_SIZE))
        surface.fill(color)
        pygame.draw.rect(surface, COLORS['WHITE'], surface.get_rect(), 1)
        if text:
            text_surf = self.item_font.render(text, True, COLORS['WHITE'])
            text_rect = text_surf.get_rect(center=surface.get_rect().center)
            surface.blit(text_surf, text_rect)
        return surface

    def _load_item_images(self):
        self.item_images = {}
        self.using_image_assets = True
        item_details = {
            "Bag": {"color": COLORS['BROWN'], "text": "B"},
            "Rope": {"color": (210, 180, 140), "text": "R"},
            "Iron Key": {"color": COLORS['DARK_GRAY'], "text": "K"},
            "Wooden Plank": {"color": COLORS['PLANK'], "text": "P"},
            "Bridge": {"color": COLORS['BRIDGE'], "text": ""},
            "Silver Key": {"color": COLORS['LIGHT_GRAY'], "text": "S"},
            "Golden Key": {"color": COLORS['YELLOW'], "text": "G"},
            "Door": {"color": (101, 67, 33), "text": "D"},
            "Slippery Tile": {"color": COLORS['CYAN'], "text": ""},
            "Tile": {"color": COLORS['GRAY'], "text": ""},
            "Tunnel": {"color": COLORS['PURPLE'], "text": ""},
            "Wall": {"color": COLORS['BLACK'], "text": ""}
        }

        for item_name, details in item_details.items():
            path = os.path.join("images", f"{item_name}.png")
            try:
                image = pygame.image.load(path).convert_alpha()
                self.item_images[item_name] = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
            except (FileNotFoundError, pygame.error):
                print(f"Warning: Image for '{item_name}' not found. Using fallback.")
                if self.using_image_assets: # Only set to false once
                    self.using_image_assets = False
                self.item_images[item_name] = self._create_fallback_surface(details['color'], details['text'])

    def _setup_rooms_and_agents(self):
        self.rooms = [
            (FirstEscapeRoom, DynamicProgrammingAgent),
            (SecondEscapeRoom, SarsaAgent),
            (ThirdEscapeRoom, QLearningAgent)
        ]
        self.current_room_index = 0

    def load_room(self, room_index):
        self.current_room_index = room_index % len(self.rooms)
        self.RoomClass, self.AgentClass = self.rooms[self.current_room_index]
        
        self.env = self.RoomClass(size=GRID_SIZE)
        self.editor_settings = {}
        all_options = {**self.env.get_editor_options(), **self.AgentClass.get_editor_options()}
        for key, details in all_options.items():
            self.editor_settings[key] = details['default']

        self._generate_new_map(log=False) 
        self.ui_manager.setup_buttons(self.agent.name)

        # Set training delay based on the room type
        if isinstance(self.env, FirstEscapeRoom):
            self.training_delay = 500
        else:
            self.training_delay = 100 # Faster for SARSA and Q-Learning

        pygame.display.set_caption(f"RL Playground - {self.agent.name}")
        self.log_message(f"Loaded {self.env.name}.")
        
        if isinstance(self.env, FirstEscapeRoom):
            self.log_message("Objective: Find the optimal path using Policy Iteration.")
        elif isinstance(self.env, SecondEscapeRoom):
            self.log_message("Objective: Evade the enemy, find the key, and escape.")
            self.log_message("Use the portal for a shortcut if it helps!")
        elif isinstance(self.env, ThirdEscapeRoom):
            self.log_message("Objective: Use planks to bridge islands, get both keys,")
            self.log_message("and unlock the door to escape.")

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.console_logs.append(f"[{timestamp}] {message}")
        line_height = 20
        max_visible_lines = self.console_rect.height // line_height
        if len(self.console_logs) > max_visible_lines:
             self.console_scroll_offset_y = (len(self.console_logs) - max_visible_lines) * line_height

    def _generate_new_map(self, log=True):
        self.is_animating = False
        self.is_paused = False
        self.is_slow_training = False
        self.is_training_paused = False
        self.death_animation_sequence = None
        
        self.env.generate_layout(self.editor_settings)
        self.agent = self.AgentClass(self.env, self.editor_settings)

        start_pos_coords = self.env.start_pos
        self.hero_sprite.rect.topleft = (start_pos_coords[1] * CELL_SIZE, start_pos_coords[0] * CELL_SIZE)
        self.hero_sprite.set_state('idle')
        
        if hasattr(self.env, 'enemy_pos') and self.env.enemy_pos:
            self.prev_enemy_pos = self.env.enemy_pos
            enemy_pos_coords = self.env.enemy_pos
            self.enemy_sprite.rect.topleft = (enemy_pos_coords[1] * CELL_SIZE, enemy_pos_coords[0] * CELL_SIZE)
            self.enemy_sprite.set_state('idle')
        
        if log:
            self.log_message("New map generated. Agent reset.")

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._draw_all()
            self.clock.tick(FPS)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()

            if self.show_skip_episode_input:
                self._handle_skip_input_events(event)
            elif self.success_popup_active:
                self._handle_popup_events(event)
            else:
                self._handle_main_events(event)

    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

    def _update(self):
        if self.death_animation_sequence:
            self._update_death_animation()
        elif self.is_animating and not self.is_paused:
            now = pygame.time.get_ticks()
            if now - self.animation_timer > self.animation_delay:
                self.animation_timer = now
                self._update_animation_step()
        
        if self.is_slow_training and not self.is_training_paused:
            now = pygame.time.get_ticks()
            if now - self.training_timer > self.training_delay:
                self.training_timer = now
                self._update_slow_train_step()
        
        self.hero_sprite.update()
        
        if hasattr(self.env, 'enemy_pos') and self.env.enemy_pos:
            current_enemy_pos = self.env.enemy_pos
            self.enemy_sprite.rect.topleft = (current_enemy_pos[1] * CELL_SIZE, current_enemy_pos[0] * CELL_SIZE)

            if self.enemy_sprite.state != 'attack':
                dx = current_enemy_pos[1] - self.prev_enemy_pos[1]
                if dx > 0: self.enemy_sprite.flip = False
                elif dx < 0: self.enemy_sprite.flip = True
                
                if self.is_animating or self.is_slow_training:
                    self.enemy_sprite.set_state('walk')
                else:
                    self.enemy_sprite.set_state('idle')

            self.enemy_sprite.update()
            self.prev_enemy_pos = current_enemy_pos

    def _update_death_animation(self):
        if self.death_animation_sequence == 'enemy_attacking':
            attack_anim = self.enemy_sprite.animations.get('attack', [])
            if self.enemy_sprite.current_frame >= len(attack_anim) - 1:
                self.hero_sprite.set_state('dead')
                self.death_animation_sequence = 'hero_dying'
        
        elif self.death_animation_sequence == 'hero_dying':
            dead_anim = self.hero_sprite.animations.get('dead', [])
            if self.hero_sprite.current_frame >= len(dead_anim) - 1:
                self.death_animation_sequence = None
    
    def _handle_main_events(self, event):
        if event.type == pygame.MOUSEWHEEL: self._handle_console_scroll(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: self._handle_click(event.pos)

    def _handle_console_scroll(self, event):
        current_console_width = self.screen.get_width() - GRID_WIDTH
        self.console_rect.width = current_console_width
        
        if not self.console_rect.collidepoint(pygame.mouse.get_pos()): return
        self.console_scroll_offset_y -= event.y * 20
        line_height = 20
        max_visible_lines = self.console_rect.height // line_height
        max_scroll = max(0, (len(self.console_logs) - max_visible_lines) * line_height)
        self.console_scroll_offset_y = max(0, min(self.console_scroll_offset_y, max_scroll))

    def _handle_click(self, pos):
        if self.death_animation_sequence: return
        for key, button_data in self.ui_manager.buttons.items():
            if button_data['visible'] and button_data['rect'].collidepoint(pos):
                handler_method = getattr(self, f"_handle_{key}_button", None)
                if handler_method:
                    handler_method()
                    return

    def _handle_refresh_button(self): self._generate_new_map()
    def _handle_edit_button(self): self._open_editor()
    def _handle_fast_train_button(self): self._run_fast_training()
    def _handle_plot_button(self): self._plot_learning_progress()
    def _handle_q_values_button(self): self.show_q_values = not self.show_q_values
    
    def _handle_slow_train_button(self):
        if not self.is_slow_training:
            self.agent.reset()
            self.is_slow_training = True
            self.is_training_paused = False
            self.training_iteration = 0
            self.log_message("Slow training started.")
        else:
            self.is_training_paused = not self.is_training_paused
            self.log_message("Slow training Paused." if self.is_training_paused else "Slow training Continued.")

    def _handle_run_button(self):
        if self.is_animating:
            self.is_paused = not self.is_paused
        elif self.agent.is_trained:
            self._start_path_animation()
        else:
            self.log_message("Agent is not trained yet!")

    def _handle_reset_run_button(self):
        if self.is_animating:
            self.log_message("Run reset.")
            self._start_path_animation()

    def _handle_skip_to_button(self):
        if self.is_slow_training and self.is_training_paused:
            self.show_skip_episode_input = True
            self.skip_episode_input_text = ""
        else:
            self.log_message("Skip To... only available during paused slow training.")

    def _open_editor(self):
        all_options = {**self.env.get_editor_options(), **self.AgentClass.get_editor_options()}
        editor = EditorMenu(self.screen, self.editor_font, self.button_font, COLORS, all_options, self.editor_settings)
        new_settings = editor.run()

        if new_settings is not None:
            self.editor_settings = new_settings
            self.log_message("Settings changed.")
            self._generate_new_map()
        else:
            self.log_message("Edit cancelled.")

    def _plot_learning_progress(self):
        if self.agent.training_type != 'episodic' or not hasattr(self.agent, 'episode_rewards') or not self.agent.episode_rewards:
            self.log_message("No plotting data available for this agent.")
            return

        plot_data = [
            {'type': 'rewards', 'data': self.agent.episode_rewards},
            {'type': 'steps', 'data': self.agent.episode_steps}
        ]
        
        if hasattr(self.agent, 'action_counts'):
            action_counts_dict = {k: dict(v) for k, v in self.agent.action_counts.items()}
            plot_data.append({'type': 'action_schema', 'data': action_counts_dict})

        plot_process = multiprocessing.Process(target=show_plots, args=(self.agent.name, self.env.grid, plot_data))
        plot_process.start()

    def _draw_all(self):
        self.screen.fill(COLORS['WHITE'])
        
        status = ""
        current_episode = getattr(self.agent, 'training_episode_count', self.training_iteration)
        if self.death_animation_sequence: status = "Caught!"
        elif self.is_animating: status = f"Animating... Step {self.animation_step}"
        elif self.is_slow_training: 
            status = f"Slow Training... {'Iteration' if self.agent.training_type == 'iterative' else 'Episode'} {self.training_iteration if self.agent.training_type == 'iterative' else current_episode}"
        if self.is_paused: status = "Run Paused"
        elif self.is_training_paused: status = "Training Paused"
        
        self._draw_grid()
        self._draw_policy()

        if self.show_q_values: self._draw_q_values()
        
        if self.hero_sprite.state != 'dead' or self.death_animation_sequence:
             self.hero_sprite.draw(self.screen)

        if hasattr(self.env, 'enemy_pos') and self.env.enemy_pos:
            self.enemy_sprite.draw(self.screen)

        self._draw_console()
        self._draw_bottom_panel(status)
        
        if self.success_popup_active: self._draw_success_popup()
        if self.show_skip_episode_input: self._draw_skip_input_box()
        
        pygame.display.flip()

    def _draw_grid(self):
        has_bag, has_rope, has_key = 0, 0, 0
        has_silver, has_golden = 0, 0
        plank1_pos, plank2_pos = None, None
        bridge1_pos, bridge2_pos = None, None
        plank_pos = None 

        state_source = None
        if self.is_animating: state_source = self.animation_state
        elif self.is_slow_training and hasattr(self.agent, 'slow_train_path') and self.agent.slow_train_path:
            state_source = self.agent.slow_train_path[-1]

        if isinstance(self.env, ThirdEscapeRoom):
            if hasattr(self.env, 'original_plank1_pos'): 
                if state_source:
                    p1_r, p1_c, p2_r, p2_c = state_source[2:6]
                    has_silver, has_golden = state_source[6:8]
                    
                    if p1_r == -1: bridge1_pos = self.env.decode_bridge_pos(p1_c)
                    else: plank1_pos = (p1_r, p1_c)
                    
                    if p2_r == -1: bridge2_pos = self.env.decode_bridge_pos(p2_c)
                    else: plank2_pos = (p2_r, p2_c)
                else: 
                    plank1_pos = self.env.original_plank1_pos
                    plank2_pos = self.env.original_plank2_pos
            else:
                if state_source:
                    plank_pos = (state_source[2], state_source[3])
                else:
                    plank_pos = self.env.original_plank_pos
                if plank_pos == (-1, -1):
                    bridge1_pos = self.env.pothole_pos
                    plank_pos = None
        
        elif state_source and len(state_source) > 2:
            if isinstance(self.agent, DynamicProgrammingAgent): 
                has_bag, has_rope = state_source[2], state_source[3]
            elif isinstance(self.agent, (SarsaAgent, QLearningAgent)): 
                has_key = state_source[2]
        
        for r, c in np.ndindex(self.env.grid.shape):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell_type = self.env.grid[r, c]

            # Draw base tile first
            self.screen.blit(self.item_images["Tile"], rect)

            if (r, c) == bridge1_pos or (r, c) == bridge2_pos:
                cell_type = BRIDGE
            
            # Draw special tiles on top of the base tile
            if cell_type == WALL:
                self.screen.blit(self.item_images["Wall"], rect)
            elif cell_type == SLIPPERY:
                self.screen.blit(self.item_images["Slippery Tile"], rect)
            elif cell_type == PORTAL:
                self.screen.blit(self.item_images["Tunnel"], rect)
            elif cell_type == POTHOLE:
                 pygame.draw.rect(self.screen, COLORS['POTHOLE'], rect) # Keep color for potholes
            elif cell_type == EXIT:
                 pygame.draw.rect(self.screen, COLORS['RED'], rect)
            elif cell_type == START:
                 pygame.draw.rect(self.screen, COLORS['GREEN'], rect)

            # Draw items on top of tiles
            if cell_type == BRIDGE:
                self.screen.blit(self.item_images["Bridge"], rect)
            if cell_type == SILVER_KEY and not has_silver:
                self.screen.blit(self.item_images["Silver Key"], rect)
            if cell_type == GOLDEN_KEY and not has_golden:
                self.screen.blit(self.item_images["Golden Key"], rect)
            if cell_type == LOCKED_DOOR and not (has_silver and has_golden):
                self.screen.blit(self.item_images["Door"], rect)

            item_map = { 5: "Bag", 6: "Rope", 7: "Iron Key" }
            if cell_type in item_map:
                if (cell_type == 5 and has_bag) or (cell_type == 6 and has_rope) or (cell_type == 7 and has_key):
                    continue
                self.screen.blit(self.item_images[item_map[cell_type]], rect)
            
            if cell_type == 4 and hasattr(self.env, 'slippery_probabilities'):
                # For SARSA/Q-Learning agents, hide slip percentages when showing Q-values,
                # as the Q-values will be drawn over them.
                if isinstance(self.agent, (SarsaAgent, QLearningAgent)):
                    if not self.show_q_values:
                        self._draw_slippery_probs(c, r)
                else: # For other agents (like DP), always show them
                    self._draw_slippery_probs(c, r)
            
            # Draw border last to frame the cell
            pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], rect, 1)
        
        plank_image = self.item_images.get("Wooden Plank")
        if plank_image:
            if plank1_pos:
                plank_rect = pygame.Rect(plank1_pos[1] * CELL_SIZE, plank1_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.screen.blit(plank_image, plank_rect)
            if plank2_pos:
                plank_rect = pygame.Rect(plank2_pos[1] * CELL_SIZE, plank2_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.screen.blit(plank_image, plank_rect)
            if plank_pos:
                plank_rect = pygame.Rect(plank_pos[1] * CELL_SIZE, plank_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                self.screen.blit(plank_image, plank_rect)

    def _draw_text_with_outline(self, text, font, pos, text_color, outline_color):
        """Helper function to draw text with a simple outline."""
        text_surf = font.render(text, True, outline_color)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.screen.blit(text_surf, (pos[0] + dx, pos[1] + dy))
        text_surf = font.render(text, True, text_color)
        self.screen.blit(text_surf, pos)

    def _draw_slippery_probs(self, c, r):
        if not hasattr(self.env, 'slippery_probabilities'): return
        probs = self.env.slippery_probabilities.get((r, c))
        if not probs: return
        
        font = self.prob_font
        
        positions = { 'up': {'midtop': (c*CELL_SIZE+CELL_SIZE/2, r*CELL_SIZE+5)}, 'down': {'midbottom': (c*CELL_SIZE+CELL_SIZE/2, (r+1)*CELL_SIZE-5)}, 'left': {'midleft': (c*CELL_SIZE+5, r*CELL_SIZE+CELL_SIZE/2)}, 'right': {'midright': ((c+1)*CELL_SIZE-5, r*CELL_SIZE+CELL_SIZE/2)} }
        for action, pos_dict in positions.items():
            prob_val = probs.get(action, 0)
            if prob_val > 0:
                text = f"{prob_val*100:.0f}%"
                text_surf = font.render(text, True, COLORS['BLACK']) # Pre-render to get rect
                text_rect = text_surf.get_rect(**pos_dict)
                
                if self.using_image_assets:
                    self._draw_text_with_outline(text, font, text_rect.topleft, COLORS['WHITE'], COLORS['BLACK'])
                else:
                    self.screen.blit(text_surf, text_rect)

    def _draw_q_values(self):
        if not hasattr(self.agent, 'q_table'): return
        
        font = self.q_value_font
        
        state_source = None
        if self.is_animating: state_source = self.animation_state
        elif self.is_slow_training and hasattr(self.agent, 'slow_train_path') and self.agent.slow_train_path:
            state_source = self.agent.slow_train_path[-1]
        
        for r, c in np.ndindex(self.env.grid.shape):
            if self.env.grid[r,c] in [1,3]: continue
            
            state_key = None
            if isinstance(self.env, ThirdEscapeRoom):
                if hasattr(self.env, 'original_plank1_pos'):
                    if state_source:
                        state_key = (r, c, *state_source[2:])
                    else: 
                        state_key = (r, c, *self.env.original_plank1_pos, *self.env.original_plank2_pos, 0, 0)
                else:
                    if state_source:
                        state_key = (r, c, state_source[2], state_source[3], state_source[4])
                    else:
                        state_key = (r, c, *self.env.original_plank_pos, 0)
            else:
                has_key_context = state_source[2] if state_source and len(state_source) > 2 else 0
                state_key = (r, c, has_key_context)

            q_vals = self.agent.q_table.get(state_key)
            if not q_vals: continue
            
            margin = 10
            positions = { 'up': {'center': (c*CELL_SIZE+CELL_SIZE/2, r*CELL_SIZE+margin)}, 'down': {'center': (c*CELL_SIZE+CELL_SIZE/2, (r+1)*CELL_SIZE-margin)}, 'left': {'center': (c*CELL_SIZE+margin, r*CELL_SIZE+CELL_SIZE/2)}, 'right': {'center': ((c+1)*CELL_SIZE-margin, r*CELL_SIZE+CELL_SIZE/2)} }
            for action, pos_dict in positions.items():
                text = f"{q_vals.get(action, 0):.1f}"
                text_surf = font.render(text, True, COLORS['BLACK']) # Pre-render to get rect
                text_rect = text_surf.get_rect(**pos_dict)
                
                if self.using_image_assets:
                    self._draw_text_with_outline(text, font, text_rect.topleft, COLORS['WHITE'], COLORS['BLACK'])
                else:
                    self.screen.blit(text_surf, text_rect)

    def _draw_policy(self):
        # This condition is changed to always draw the policy for the DP agent
        if not isinstance(self.agent, DynamicProgrammingAgent):
            return

        policy_context = {}
        state_source = self.animation_state if self.is_animating else None
        
        if isinstance(self.agent, DynamicProgrammingAgent):
            if state_source:
                policy_context['has_bag'] = state_source[2]
                policy_context['has_rope'] = state_source[3]
            else:
                start_items = self.editor_settings.get("Start with Items", "None")
                policy_context['has_bag'] = 1 if start_items in ["Bag", "Both"] else 0
                policy_context['has_rope'] = 1 if start_items in ["Rope", "Both"] else 0

            for r, c in np.ndindex(self.env.grid.shape):
                if self.env.grid[r, c] == WALL: continue

                if self.env.grid[r, c] == SLIPPERY:
                    best_next_pos = None
                    max_val = -np.inf
                    
                    tile_probs = self.env.slippery_probabilities.get((r, c), {})
                    for action, prob in tile_probs.items():
                        if prob > 0:
                            dr, dc = self.env.action_space[action]
                            next_r, next_c = r + dr, c + dc
                            
                            val = 0
                            is_bag_tile = hasattr(self.env, 'bag_pos') and self.env.bag_pos and (next_r, next_c) == self.env.bag_pos
                            is_rope_tile = hasattr(self.env, 'rope_pos') and self.env.rope_pos and (next_r, next_c) == self.env.rope_pos
                            
                            if is_bag_tile and not policy_context['has_bag']:
                                reward = 20
                                next_state_after_pickup = (next_r, next_c, 1, policy_context['has_rope'])
                                val = reward + self.agent.gamma * self.agent.value_function[next_state_after_pickup]
                            elif is_rope_tile and not policy_context['has_rope']:
                                reward = 30 if policy_context['has_bag'] else -10
                                next_state_after_pickup = (next_r, next_c, policy_context['has_bag'], 1)
                                val = reward + self.agent.gamma * self.agent.value_function[next_state_after_pickup]
                            else:
                                next_state_key = (next_r, next_c, policy_context['has_bag'], policy_context['has_rope'])
                                val = self.agent.value_function[next_state_key]

                            if val > max_val:
                                max_val = val
                                best_next_pos = (next_r, next_c)

                    if best_next_pos:
                        center_x, center_y = c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2
                        best_center_x = best_next_pos[1] * CELL_SIZE + CELL_SIZE / 2
                        best_center_y = best_next_pos[0] * CELL_SIZE + CELL_SIZE / 2
                        
                        dx, dy = best_center_x - center_x, best_center_y - center_y
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 0:
                            arrow_len = CELL_SIZE * 0.2
                            end_x = center_x + (dx / dist) * arrow_len
                            end_y = center_y + (dy / dist) * arrow_len
                            self._draw_arrow((center_x, center_y), (end_x, end_y))

                else: 
                    state_key = (r, c, policy_context['has_bag'], policy_context['has_rope'])
                    action = self.agent.policy[state_key]
                    if action is None: continue
                    center_x, center_y = c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2
                    arrow_len = CELL_SIZE * 0.2
                    end_pos_delta = {'up':(0,-1),'down':(0,1),'left':(-1,0),'right':(1,0)}.get(action)
                    if end_pos_delta:
                        self._draw_arrow((center_x, center_y), (center_x + end_pos_delta[0]*arrow_len, center_y + end_pos_delta[1]*arrow_len))

    def _draw_arrow(self, start, end):
        pygame.draw.line(self.screen, COLORS['BLACK'], start, end, 3)
        rotation = np.degrees(np.arctan2(start[1]-end[1], end[0]-start[0])) + 90
        pygame.draw.polygon(self.screen, COLORS['BLACK'], ((end[0]+8*np.sin(np.radians(rotation)), end[1]+8*np.cos(np.radians(rotation))), (end[0]+8*np.sin(np.radians(rotation-120)), end[1]+8*np.cos(np.radians(rotation-120))), (end[0]+8*np.sin(np.radians(rotation+120)), end[1]+8*np.cos(np.radians(rotation+120)))))

    def _draw_console(self):
        current_console_width = self.screen.get_width() - GRID_WIDTH
        self.console_rect = pygame.Rect(GRID_WIDTH, 0, current_console_width, GRID_WIDTH)
        
        pygame.draw.rect(self.screen, COLORS['CONSOLE_BG'], self.console_rect)
        line_height = 20
        y_offset = 10 - (self.console_scroll_offset_y % line_height)
        start_index = self.console_scroll_offset_y // line_height
        
        for i in range(start_index, len(self.console_logs)):
            log = self.console_logs[i];
            if y_offset > self.console_rect.height - 20: break
            text_surf = self.console_font.render(log, True, COLORS['CONSOLE_TEXT']);
            self.screen.blit(text_surf, (self.console_rect.x + 10, y_offset));
            y_offset += line_height
    
    def _draw_bottom_panel(self, status_text):
        if status_text:
            text_surf = self.status_font.render(status_text, True, COLORS['BLACK'])
            self.screen.blit(text_surf, text_surf.get_rect(center=(GRID_WIDTH/2, GRID_WIDTH + 20)))
        
        self.ui_manager.update_button_text({
            'is_paused': self.is_paused, 'is_animating': self.is_animating,
            'is_training_paused': self.is_training_paused, 'is_slow_training': self.is_slow_training,
            'show_q_values': self.show_q_values
        })
        self.ui_manager.draw()

    def _run_fast_training(self):
        if not self.is_slow_training:
            self.agent.reset()
        self.log_message(f"Fast training {self.agent.name}...")
        
        if self.agent.training_type == 'iterative':
            max_iter = 500
            for i in range(1, max_iter + 1):
                converged, delta = self.agent.train_step()
                if converged:
                    self.log_message(f"Converged after {i} iterations."); break
        else:
            max_episodes = self.agent.max_episodes
            current_episode = getattr(self.agent, 'training_episode_count', 0)
            num_episodes_to_run = max_episodes - current_episode
            for i in range(num_episodes_to_run):
                self.agent.train_step()
        
        self.log_message(f"Training finished."); self.agent.extract_policy()
        self.is_slow_training = False
        self.is_training_paused = False

    def _update_slow_train_step(self):
        if self.agent.training_type == 'iterative':
            self.training_iteration += 1
            converged, delta = self.agent.train_step()
            self.agent.extract_policy()
            if converged:
                self.log_message(f"DP converged in {self.training_iteration} iterations.")
                self.is_training_paused = True
        else:
            if hasattr(self.agent, 'train_step_by_step'):
                 _, self.agent.slow_train_path = self.agent.train_step_by_step()
                 if not self.agent.slow_train_episode_active:
                      self.log_message(f"  ... episode {self.agent.training_episode_count} finished.")
                 self.agent.extract_policy()
                 if self.agent.slow_train_path:
                     pos = self.agent.slow_train_path[-1][:2]
                     self.hero_sprite.rect.topleft = (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE)
                     self.hero_sprite.set_state('idle')
            else:
                 self.log_message("This agent does not support step-by-step training.", "WARNING")

    def _start_path_animation(self):
        self.is_animating = True; self.is_paused = False; self.env.reset_state()
        self.death_animation_sequence = None

        if hasattr(self.env, 'enemy_pos') and self.env.enemy_pos:
            self.prev_enemy_pos = self.env.enemy_pos
            self.enemy_sprite.set_state('idle')
        
        if isinstance(self.env, ThirdEscapeRoom):
            self.animation_state = (*self.env.start_pos, 
                                    *self.env.original_plank1_pos, 
                                    *self.env.original_plank2_pos, 
                                    0, 0)
        elif isinstance(self.agent, DynamicProgrammingAgent):
            start_items = self.editor_settings.get("Start with Items", "None")
            start_bag = 1 if start_items in ["Bag", "Both"] else 0
            start_rope = 1 if start_items in ["Rope", "Both"] else 0
            self.animation_state = (*self.env.start_pos, start_bag, start_rope)
        elif isinstance(self.agent, (SarsaAgent, QLearningAgent)):
            self.animation_state = (*self.env.start_pos, 0)
        
        pos = self.env.start_pos
        self.hero_sprite.rect.topleft = (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE)
        self.hero_sprite.set_state('run')

        self.animation_step = 0; self.log_message("Run started.")
        self.animation_total_reward = 0

    def _update_animation_step(self):
        if isinstance(self.agent, DynamicProgrammingAgent):
            action = self.agent.policy[self.animation_state]
        else:
            action = self.agent.policy.get(self.animation_state)

        if action is None:
            self._handle_run_end(False, "No policy for state")
            return

        prev_pos = self.animation_state[:2]
        next_state, reward, done = self.env.step(self.animation_state, action)
        self.animation_total_reward += (self.agent.gamma ** self.animation_step) * reward
        self.animation_state = next_state
        self.animation_step += 1

        current_pos = self.animation_state[:2]
        self.hero_sprite.rect.topleft = (current_pos[1] * CELL_SIZE, current_pos[0] * CELL_SIZE)
        
        dx = current_pos[1] - prev_pos[1]
        if dx > 0: self.hero_sprite.flip = False
        elif dx < 0: self.hero_sprite.flip = True
        
        if self.env.get_state_type(current_pos) == 4:
             self.hero_sprite.set_state('slide')
        elif self.hero_sprite.state == 'slide':
             self.hero_sprite.set_state('run')
        
        if done:
            success = reward > 0
            reason = "Reached Terminal State"
            if (isinstance(self.agent, SarsaAgent) or isinstance(self.agent, QLearningAgent)) and not success:
                reason = "Agent was caught!"
            self._handle_run_end(success, reason)
        elif self.animation_step >= 200:
            self._handle_run_end(False, "Max steps reached")

    def _handle_run_end(self, success, reason):
        self.is_animating = False
        if success:
            self.log_message(f"Run Successful! {reason}. Steps: {self.animation_step}, Reward: {self.animation_total_reward:.2f}")
            self.hero_sprite.set_state('idle')
            self.success_popup_active = True
        else:
            self.log_message(f"Run Failed: {reason}. Steps: {self.animation_step}, Reward: {self.animation_total_reward:.2f}")
            if hasattr(self.env, 'enemy_pos') and self.env.enemy_pos and reason == "Agent was caught!":
                self.death_animation_sequence = 'enemy_attacking'
                self.enemy_sprite.set_state('attack')
            else:
                self.hero_sprite.set_state('dead')

    def _handle_popup_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.next_room_button_rect.collidepoint(event.pos): self.success_popup_active = False; self.load_room(self.current_room_index + 1)
            elif self.stay_here_button_rect.collidepoint(event.pos): self.success_popup_active = False

    def _draw_success_popup(self):
        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA); overlay.fill((0,0,0,128)); self.screen.blit(overlay, (0,0))
        popup_rect = pygame.Rect((screen_w-400)/2, (screen_h-200)/2, 400, 200)
        pygame.draw.rect(self.screen, COLORS['LIGHT_GRAY'], popup_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], popup_rect, 3, border_radius=15)
        title_surf = self.popup_font.render("Success!", True, COLORS['BLACK']); self.screen.blit(title_surf, title_surf.get_rect(center=(popup_rect.centerx, popup_rect.top + 40)))
        self.next_room_button_rect = pygame.Rect(popup_rect.left+50, popup_rect.bottom-70, 140, 50)
        self.stay_here_button_rect = pygame.Rect(popup_rect.right-190, popup_rect.bottom-70, 140, 50)
        pygame.draw.rect(self.screen, COLORS['GREEN'], self.next_room_button_rect, border_radius=8); self.screen.blit(self.button_font.render("Next Room", True, COLORS['WHITE']), self.next_room_button_rect.inflate(-20,-10))
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], self.stay_here_button_rect, border_radius=8); self.screen.blit(self.button_font.render("Another Round", True, COLORS['WHITE']), self.stay_here_button_rect.inflate(-20,-10))

    def _handle_skip_input_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                try:
                    target_episode = int(self.skip_episode_input_text)
                    current_episode = getattr(self.agent, 'training_episode_count', 0)
                    max_episodes = self.agent.max_episodes
                    if current_episode < target_episode <= max_episodes:
                        self._run_skip_training(target_episode)
                    else:
                        self.log_message(f"Invalid episode. Enter a number between {current_episode+1} and {max_episodes}.")
                except ValueError:
                    self.log_message("Invalid input. Please enter a number.")
                self.show_skip_episode_input = False
                self.skip_episode_input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.skip_episode_input_text = self.skip_episode_input_text[:-1]
            elif event.unicode.isdigit():
                self.skip_episode_input_text += event.unicode
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.show_skip_episode_input = False
            self.skip_episode_input_text = ""

    def _draw_skip_input_box(self):
        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA); overlay.fill((0,0,0,128)); self.screen.blit(overlay, (0,0))
        box_width, box_height = 400, 150
        box_rect = pygame.Rect((screen_w-box_width)/2, (screen_h-box_height)/2, box_width, box_height)
        pygame.draw.rect(self.screen, COLORS['LIGHT_GRAY'], box_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], box_rect, 3, border_radius=15)
        
        prompt_surf = self.editor_font.render("Skip to episode (press Enter):", True, COLORS['BLACK'])
        self.screen.blit(prompt_surf, (box_rect.x + 20, box_rect.y + 20))
        
        input_rect = pygame.Rect(box_rect.x + 20, box_rect.y + 60, box_width - 40, 40)
        pygame.draw.rect(self.screen, COLORS['WHITE'], input_rect)
        pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], input_rect, 2)
        
        input_surf = self.editor_font.render(self.skip_episode_input_text, True, COLORS['BLACK'])
        self.screen.blit(input_surf, (input_rect.x + 10, input_rect.y + 5))

    def _run_skip_training(self, target_episode):
        current_episode = getattr(self.agent, 'training_episode_count', 0)
        num_episodes_to_run = target_episode - current_episode
        if num_episodes_to_run <= 0:
            self.log_message("Target episode is not in the future.")
            return

        self.log_message(f"Skipping training to episode {target_episode}...")
        for i in range(num_episodes_to_run):
            self.agent.train_step()
        
        self.agent.extract_policy()
        if hasattr(self.agent, 'slow_train_episode_active'):
            self.agent.slow_train_episode_active = False

        if hasattr(self.agent, 'slow_train_path'):
            if isinstance(self.env, ThirdEscapeRoom):
                if hasattr(self.env, 'original_plank1_pos'): 
                    initial_state = (*self.env.start_pos, 
                                     *self.env.original_plank1_pos, 
                                     *self.env.original_plank2_pos, 
                                     0, 0)
                else: 
                    initial_state = (*self.env.start_pos, *self.env.original_plank_pos, 0)
            elif isinstance(self.agent, DynamicProgrammingAgent):
                start_items = self.editor_settings.get("Start with Items", "None")
                start_bag = 1 if start_items in ["Bag", "Both"] else 0
                start_rope = 1 if start_items in ["Rope", "Both"] else 0
                initial_state = (*self.env.start_pos, start_bag, start_rope)
            elif isinstance(self.agent, (SarsaAgent, QLearningAgent)):
                initial_state = (*self.env.start_pos, 0)
            
            self.agent.slow_train_path = [initial_state]
            
            start_pos_coords = self.env.start_pos
            self.hero_sprite.rect.topleft = (start_pos_coords[1] * CELL_SIZE, start_pos_coords[0] * CELL_SIZE)
            self.hero_sprite.set_state('idle')
        
        self.log_message(f"Training skipped to episode {self.agent.training_episode_count}.")
        self.is_training_paused = True 

if __name__ == "__main__":
    multiprocessing.freeze_support()
    visualizer = PygameVisualizer()
    visualizer.run()

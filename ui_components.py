import pygame

class Dropdown:
    """A reusable, scrollable dropdown menu component with fixes for scrolling."""
    def __init__(self, rect, options, font, colors, selected_option=None):
        self.rect = pygame.Rect(rect)
        self.options = [str(o) for o in options]
        self.font = font
        self.colors = colors
        self.selected_option = selected_option if selected_option in self.options else (self.options[0] if self.options else "")
        self.is_open = False
        self.scroll_offset = 0
        self.max_visible_items = 6

    def handle_event(self, event, mouse_pos):
        """
        Handles mouse events for the dropdown.
        Returns True if the event was consumed, False otherwise.
        """
        if event.type == pygame.MOUSEWHEEL and self.is_open:
            # Pass scroll events to the dedicated handler
            return self.handle_scroll(event, mouse_pos)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(mouse_pos):
                self.is_open = not self.is_open
                return True  # Consumed the click to open/close

            if self.is_open:
                # If open, check if an option was clicked
                clicked_option = self.check_option_click(mouse_pos)
                if clicked_option is not None:
                    self.selected_option = clicked_option
                # Any click outside the main box closes the dropdown
                self.is_open = False
                return True  # Consumed the click

        return False # Event was not handled by this dropdown

    def handle_scroll(self, event, mouse_pos):
        """Handles mouse wheel scrolling logic."""
        if not self.is_open:
            return False

        # Define the area where scrolling is active (the list of options)
        visible_options = self.get_visible_options()
        options_height = len(visible_options) * self.rect.height
        scroll_area = pygame.Rect(self.rect.x, self.rect.bottom, self.rect.width, options_height)

        # Only scroll if the mouse is over the options list
        if scroll_area.collidepoint(mouse_pos):
            # event.y is 1 for scroll up, -1 for scroll down
            self.scroll_offset -= event.y
            
            # Clamp the scroll offset to valid bounds
            max_scroll = max(0, len(self.options) - self.max_visible_items)
            self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
            return True # Consumed the scroll event
            
        return False

    def get_visible_options(self):
        """Gets the sublist of options that should be visible based on the scroll offset."""
        start = int(self.scroll_offset)
        end = start + self.max_visible_items
        return self.options[start:end]

    def check_option_click(self, pos):
        """Checks if a click occurred on one of the visible options."""
        visible_options = self.get_visible_options()
        for i, option in enumerate(visible_options):
            option_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.height, self.rect.width, self.rect.height)
            if option_rect.collidepoint(pos):
                return option
        return None

    def draw(self, surface):
        """Draws the main (closed) dropdown box."""
        pygame.draw.rect(surface, self.colors['WHITE'], self.rect)
        pygame.draw.rect(surface, self.colors['DARK_GRAY'], self.rect, 2)
        
        text_surf = self.font.render(str(self.selected_option), True, self.colors['BLACK'])
        surface.blit(text_surf, (self.rect.x + 10, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))
        
        # Draw dropdown arrow
        arrow_points = [
            (self.rect.right - 20, self.rect.centery - 4),
            (self.rect.right - 10, self.rect.centery - 4),
            (self.rect.right - 15, self.rect.centery + 4)
        ]
        pygame.draw.polygon(surface, self.colors['DARK_GRAY'], arrow_points)
        
    def draw_options(self, surface):
        """Draws the open dropdown list. Should be called last to render on top."""
        if not self.is_open:
            return

        mouse_pos = pygame.mouse.get_pos()
        visible_options = self.get_visible_options()
        
        for i, option in enumerate(visible_options):
            option_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.height, self.rect.width, self.rect.height)
            
            # Highlight option on hover
            color = self.colors['HOVER_BLUE'] if option_rect.collidepoint(mouse_pos) else self.colors['WHITE']
            pygame.draw.rect(surface, color, option_rect)
            pygame.draw.rect(surface, self.colors['DARK_GRAY'], option_rect, 1)
            
            text_surf = self.font.render(str(option), True, self.colors['BLACK'])
            surface.blit(text_surf, (option_rect.x + 10, option_rect.y + (option_rect.height - text_surf.get_height()) // 2))


class InputBox:
    """A reusable text input box component with validation."""
    def __init__(self, rect, text, font, colors, input_type='text', default_value=None):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.colors = colors
        self.active = False
        self.input_type = input_type
        self.default_value = default_value if default_value is not None else text
        self.text = str(text)

    def handle_event(self, event, mouse_pos):
        """Handles mouse clicks and keyboard input."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(mouse_pos)
            return True # Consumes click event

        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                char = event.unicode
                # Basic validation
                if self.input_type == 'int' and char.isdigit():
                    self.text += char
                elif self.input_type == 'float' and (char.isdigit() or (char in '.-e' and char not in self.text)):
                     self.text += char
                elif self.input_type == 'text':
                    self.text += char
            return True # Consumes key press
        return False

    def get_value(self):
        """Returns the sanitized value, or the default if conversion fails."""
        try:
            if self.input_type == 'int': return int(self.text)
            if self.input_type == 'float': return float(self.text)
            return self.text
        except (ValueError, TypeError):
            try:
                if self.input_type == 'int': return int(self.default_value)
                if self.input_type == 'float': return float(self.default_value)
            except (ValueError, TypeError):
                return 0 if self.input_type != 'text' else ""
            return self.default_value

    def draw(self, surface):
        """Draws the input box."""
        border_color = self.colors['BLUE'] if self.active else self.colors['DARK_GRAY']
        pygame.draw.rect(surface, self.colors['WHITE'], self.rect)
        pygame.draw.rect(surface, border_color, self.rect, 2)
        
        text_surf = self.font.render(self.text, True, self.colors['BLACK'])
        surface.blit(text_surf, (self.rect.x + 10, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

class UIManager:
    def __init__(self, screen, button_font, colors):
        self.screen = screen
        self.button_font = button_font
        self.colors = colors
        self.buttons = {}

    def setup_buttons(self, agent_type):
        self.buttons = {}
        button_keys = ["refresh", "edit", "fast_train", "slow_train", "run", "reset_run", "plot"]
        if agent_type in ["SARSA", "Q-Learning"]:
            button_keys.insert(7, "skip_to")
            button_keys.insert(8, "q_values")
        
        button_texts = {
            "refresh": "Refresh Map", "edit": "Edit Map", "fast_train": "Fast Train",
            "slow_train": "Slow Train", "run": "Run", "reset_run": "Reset Run", "q_values": "Show Q-Vals",
            "skip_to": "Skip To...", "plot": "Plot Data"
        }

        for key in button_keys:
            self.buttons[key] = {"text": button_texts[key], "rect": pygame.Rect(0, 0, 110, 50), "visible": True}

    def update_button_text(self, visualizer_state):
        self.buttons['run']['text'] = "Resume" if visualizer_state['is_paused'] else ("Pause" if visualizer_state['is_animating'] else "Run")
        self.buttons['slow_train']['text'] = "Continue" if visualizer_state['is_training_paused'] else "Pause" if visualizer_state['is_slow_training'] else "Slow Train"
        if 'q_values' in self.buttons:
            self.buttons['q_values']['text'] = "Hide Q-Vals" if visualizer_state['show_q_values'] else "Show Q-Vals"
        
        self.buttons['reset_run']['visible'] = visualizer_state['is_animating']

    def draw(self):
        current_window_width = self.screen.get_width()
        panel_rect = pygame.Rect(0, self.screen.get_height() - 100, current_window_width, 100)
        pygame.draw.rect(self.screen, self.colors['WHITE'], panel_rect)
        pygame.draw.line(self.screen, self.colors['DARK_GRAY'], (0, panel_rect.top), (current_window_width, panel_rect.top), 2)

        visible_buttons = {k: v for k, v in self.buttons.items() if v['visible']}
        num_buttons = len(visible_buttons)
        if num_buttons == 0: return

        button_width = list(visible_buttons.values())[0]['rect'].width
        total_button_width = num_buttons * button_width
        gap = (current_window_width - total_button_width) / (num_buttons + 1)
        
        start_x = gap
        button_y = panel_rect.top + (panel_rect.height - 50) / 2

        for i, (key, button_data) in enumerate(visible_buttons.items()):
            x = start_x + i * (button_width + gap)
            button_data['rect'].topleft = (x, button_y)
            
            pygame.draw.rect(self.screen, self.colors['DARK_GRAY'], button_data['rect'], border_radius=10)
            text_surf = self.button_font.render(button_data['text'], True, self.colors['WHITE'])
            self.screen.blit(text_surf, text_surf.get_rect(center=button_data['rect'].center))

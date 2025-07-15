import pygame
from ui_components import Dropdown, InputBox

class EditorMenu:
    """
    A self-contained class to handle the entire "Edit Map" menu logic.
    It runs its own event loop, making it independent from the main game loop.
    """
    def __init__(self, screen, editor_font, button_font, colors, options, current_settings):
        self.screen = screen
        self.editor_font = editor_font
        self.button_font = button_font
        self.colors = colors
        self.options = options
        self.current_settings = current_settings
        self.is_running = True
        self.new_settings = None # This will be the return value

        self.window_width, self.window_height = screen.get_size()
        self.editor_components = {}
        self._setup_ui()

    def _setup_ui(self):
        """Creates the UI components (dropdowns, input boxes) based on the provided options."""
        y_offset = 0 # Will be set properly in the draw loop
        for key, details in self.options.items():
            comp_type = details.get("type", "dropdown")
            # Use current setting if available, otherwise fall back to default
            default_value = self.current_settings.get(key, details.get("default"))
            
            comp_rect = pygame.Rect(0, 0, 250, 40) # Placeholder rect

            if comp_type == "input":
                input_type = details.get("input_type", "text")
                self.editor_components[key] = InputBox(comp_rect, default_value, self.editor_font, self.colors, input_type=input_type, default_value=details.get("default"))
            else: # Dropdown
                self.editor_components[key] = Dropdown(comp_rect, details["options"], self.editor_font, self.colors, selected_option=str(default_value))

    def run(self):
        """Runs the main loop for the editor menu. Returns the new settings or None."""
        while self.is_running:
            self._handle_events()
            self._draw()
        return self.new_settings

    def _handle_events(self):
        """
        Handles all events for the editor menu. Button clicks are checked first,
        then events are passed to UI components.
        """
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                self.new_settings = None
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.save_button_rect.collidepoint(mouse_pos):
                    self.new_settings = {}
                    for key, comp in self.editor_components.items():
                        if isinstance(comp, Dropdown):
                            self.new_settings[key] = comp.selected_option
                        elif isinstance(comp, InputBox):
                            self.new_settings[key] = comp.get_value()
                    self.is_running = False
                    return

                if self.cancel_button_rect.collidepoint(mouse_pos):
                    self.new_settings = None
                    self.is_running = False
                    return

            open_dropdown_handled_event = False
            for component in self.editor_components.values():
                if isinstance(component, Dropdown) and component.is_open:
                    component.handle_event(event, mouse_pos)
                    open_dropdown_handled_event = True
                    break

            if not open_dropdown_handled_event:
                for component in self.editor_components.values():
                    component.handle_event(event, mouse_pos)


    def _draw(self):
        """Draws the entire editor menu, ensuring correct layering."""
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        panel_width = 600
        panel_height = 150 + len(self.editor_components) * 60
        editor_rect = pygame.Rect((self.window_width - panel_width) / 2, (self.window_height - panel_height) / 2, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.colors['LIGHT_GRAY'], editor_rect, border_radius=15)
        pygame.draw.rect(self.screen, self.colors['DARK_GRAY'], editor_rect, 3, border_radius=15)

        y_offset = editor_rect.top + 50
        for key, comp in self.editor_components.items():
            label_surf = self.editor_font.render(f"{key}:", True, self.colors['BLACK'])
            self.screen.blit(label_surf, (editor_rect.x + 40, y_offset + 10))
            
            comp.rect.topleft = (editor_rect.x + 280, y_offset)
            comp.draw(self.screen)
            y_offset += 60

        button_y = editor_rect.bottom - 70
        self.save_button_rect = pygame.Rect(editor_rect.centerx - 120, button_y, 110, 50)
        self.cancel_button_rect = pygame.Rect(editor_rect.centerx + 10, button_y, 110, 50)
        
        pygame.draw.rect(self.screen, self.colors['GREEN'], self.save_button_rect, border_radius=10)
        save_text = self.button_font.render("Save", True, self.colors['WHITE'])
        self.screen.blit(save_text, save_text.get_rect(center=self.save_button_rect.center))
        
        pygame.draw.rect(self.screen, self.colors['RED'], self.cancel_button_rect, border_radius=10)
        cancel_text = self.button_font.render("Cancel", True, self.colors['WHITE'])
        self.screen.blit(cancel_text, cancel_text.get_rect(center=self.cancel_button_rect.center))

        for comp in self.editor_components.values():
            if isinstance(comp, Dropdown):
                comp.draw_options(self.screen)

        pygame.display.flip()

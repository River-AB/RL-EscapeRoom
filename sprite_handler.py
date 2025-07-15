import pygame
import os

class AnimatedSprite(pygame.sprite.Sprite):
    """
    A class to handle loading, animating, and displaying sprites
    for different characters based on a directory naming convention.
    Includes a fallback to simple colored blocks if image assets are not found.
    """
    def __init__(self, size, sprite_name):
        super().__init__()
        self.size = size
        self.sprite_name = sprite_name
        
        try:
            self.animations = self._load_animations(size)
            if not self.animations: # If loading returned empty, trigger fallback
                raise FileNotFoundError 
        except (FileNotFoundError, pygame.error):
            print(f"Warning: Asset folder for '{self.sprite_name}' not found or empty. Using colored block fallback.")
            self.animations = self._create_fallback_animations()

        self.game_states = list(self.animations.keys())

        # Set a default state, preferring 'idle' or 'walk' if available
        if 'idle' in self.animations:
            self.state = 'idle'
        elif 'walk' in self.animations:
            self.state = 'walk'
        elif self.game_states:
            self.state = self.game_states[0]
        else:
            self.state = ''

        self.current_frame = 0
        self.last_update = pygame.time.get_ticks()
        self.animation_speed = 100 # Milliseconds per frame

        if self.state and self.animations.get(self.state):
            self.image = self.animations[self.state][self.current_frame]
            self.rect = self.image.get_rect()
        else:
            # Create a blank surface if no animations are found at all
            self.image = pygame.Surface((size, size), pygame.SRCALPHA)
            self.rect = self.image.get_rect()

        self.flip = False

    def _load_animations(self, size):
        """
        Loads all animation frames from a character-specific directory.
        e.g., 'images/Hero', 'images/Enemy'
        It assumes subfolders named 'Idle', 'Run', 'Walk', 'Attack', etc.
        """
        animations = {}
        base_path = os.path.join('images', self.sprite_name)
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Sprite directory not found: {base_path}")

        animation_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

        for anim_name in animation_folders:
            anim_path = os.path.join(base_path, anim_name)
            images = []
            try:
                sorted_files = sorted(os.listdir(anim_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
            except (ValueError, IndexError):
                print(f"Warning: Could not sort files in {anim_path} numerically. Using alphabetical sort.")
                sorted_files = sorted(os.listdir(anim_path))

            for filename in sorted_files:
                if filename.endswith('.png'):
                    img_path = os.path.join(anim_path, filename)
                    image = pygame.image.load(img_path).convert_alpha()
                    image = pygame.transform.scale(image, (self.size, self.size))
                    images.append(image)

            if images:
                animations[anim_name.lower()] = images

        return animations
        
    def _create_fallback_animations(self):
        """Creates simple colored block animations as a fallback."""
        animations = {}
        colors = {
            'Hero': (0, 180, 0),   # Green
            'Enemy': (180, 0, 0)   # Red
        }
        base_color = colors.get(self.sprite_name, (100, 100, 100))
        
        anim_states = ['idle', 'walk', 'run', 'attack', 'dead', 'slide']
        for state in anim_states:
            color = base_color
            if state == 'dead':
                color = (50, 50, 50)
            elif state == 'attack':
                color = (255, 100, 0)

            surface = pygame.Surface((self.size, self.size))
            surface.fill(color)
            pygame.draw.rect(surface, (255,255,255), surface.get_rect(), 2)
            animations[state] = [surface]
        return animations

    def set_state(self, new_state):
        """Sets a new animation state, resetting the frame counter if changed."""
        if new_state in self.animations and self.state != new_state:
            self.state = new_state
            self.current_frame = 0
            self.last_update = pygame.time.get_ticks()

    def update(self):
        """Updates the animation frame based on the current state and timer."""
        if not self.state or self.state not in self.animations or not self.animations[self.state]:
            return

        now = pygame.time.get_ticks()

        if now - self.last_update > self.animation_speed:
            self.last_update = now
            self.current_frame += 1

            animation_frames = self.animations[self.state]

            if self.state in ['dead', 'slide', 'jump', 'attack']:
                if self.current_frame >= len(animation_frames):
                    self.current_frame = len(animation_frames) - 1
            else: 
                if len(animation_frames) > 0:
                    self.current_frame %= len(animation_frames)
        
        self.image = self.animations[self.state][self.current_frame]
        if self.flip:
            self.image = pygame.transform.flip(self.image, True, False)


    def draw(self, surface):
        """Draws the sprite on the given surface."""
        surface.blit(self.image, self.rect)

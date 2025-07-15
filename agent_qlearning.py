import numpy as np
import random
from collections import defaultdict
from base_classes import BaseAgent
from room3_qlearning_env import ThirdEscapeRoom

class QLearningAgent(BaseAgent):
    """An agent that learns using the Q-Learning (model-free, off-policy) algorithm."""
    def __init__(self, env, settings):
        super().__init__(env, settings)
        self.name = "Q-Learning"
        self.training_type = "episodic"
        
        self.gamma = float(settings.get('Gamma', 0.9))
        self.alpha = float(settings.get('Alpha', 0.1))
        self.epsilon = float(settings.get('Epsilon', 1.0))
        self.max_steps = int(settings.get('Max Steps', 200))
        self.max_episodes = int(settings.get('Max Episodes', 10000))

        self.epsilon_decay = float(settings.get('Epsilon Decay', 0.9995))
        self.min_epsilon = float(settings.get('Min Epsilon', 0.01))
        
        self.reset()
    
    @staticmethod
    def get_editor_options():
        """Returns agent-specific settings for the editor UI."""
        return {
            "Alpha": {"type": "input", "default": "0.1", "input_type": "float"},
            "Epsilon": {"type": "input", "default": "1.0", "input_type": "float"},
            "Gamma": {"type": "input", "default": "0.9", "input_type": "float"},
            "Epsilon Decay": {"type": "input", "default": "0.9995", "input_type": "float"},
            "Min Epsilon": {"type": "input", "default": "0.01", "input_type": "float"},
            "Max Episodes": {"type": "input", "default": "10000", "input_type": "int"},
            "Max Steps": {"type": "input", "default": "200", "input_type": "int"},
        }

    def reset(self):
        """Resets the agent's Q-table and policy for a new training session."""
        self.q_table = defaultdict(lambda: {action: 0.0 for action in self.env.action_space})
        self.policy = {}
        self.is_trained = False
        self.episode_rewards = []
        self.episode_steps = []
        self.training_episode_count = 0
        self.action_counts = defaultdict(lambda: defaultdict(int))
        
        self.slow_train_episode_active = False
        self.slow_train_state = None
        self.slow_train_action = None
        self.slow_train_path = []
        self.slow_train_step_count = 0

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        all_actions = list(self.env.action_space.keys())

        if not all_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(all_actions)
        else:
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(all_actions)
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def train_step(self):
        """
        Runs a full training episode using the Q-Learning algorithm.
        """
        self.env.reset_state()
        
        start_pos = self.env.start_pos
        if isinstance(self.env, ThirdEscapeRoom):
            if hasattr(self.env, 'original_plank1_pos'):
                state = (*start_pos, 
                         *self.env.original_plank1_pos, 
                         *self.env.original_plank2_pos, 
                         0, 0)
            else: 
                state = (*start_pos, *self.env.original_plank_pos, 0)
        else:
            state = (*start_pos, 0)
        
        done = False
        path = [state]
        step_count = 0
        total_reward = 0
        
        while not done and step_count < self.max_steps:
            action = self.choose_action(state)
            if action is None: break
            
            if action in ['up', 'down', 'left', 'right']:
                self.action_counts[state[:2]][action] += 1

            next_state, reward, done = self.env.step(state, action)
            total_reward += reward
            
            old_value = self.q_table[state][action]
            
            next_q_values = self.q_table[next_state]
            
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0

            new_value = old_value + self.alpha * (reward + self.gamma * max_next_q - old_value)
            self.q_table[state][action] = new_value
            
            state = next_state
            path.append(state)
            step_count += 1
        
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(step_count)
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.training_episode_count += 1
        return False, path

    def train_step_by_step(self):
        """
        Runs a single step of a training episode for visualization purposes.
        """
        if not self.slow_train_episode_active:
            self.env.reset_state()
            if isinstance(self.env, ThirdEscapeRoom):
                if hasattr(self.env, 'original_plank1_pos'):
                    self.slow_train_state = (*self.env.start_pos, 
                                             *self.env.original_plank1_pos, 
                                             *self.env.original_plank2_pos, 
                                             0, 0)
                else:
                    self.slow_train_state = (*self.env.start_pos, *self.env.original_plank_pos, 0)
            else:
                self.slow_train_state = (*self.env.start_pos, 0)

            self.slow_train_action = self.choose_action(self.slow_train_state)
            self.slow_train_path = [self.slow_train_state]
            self.slow_train_episode_active = True
            self.slow_train_step_count = 0

        done = False
        if self.slow_train_action is not None and self.slow_train_step_count < self.max_steps:
            if self.slow_train_action in ['up', 'down', 'left', 'right']:
                self.action_counts[self.slow_train_state[:2]][self.slow_train_action] += 1

            next_state, reward, done = self.env.step(self.slow_train_state, self.slow_train_action)
            
            old_value = self.q_table[self.slow_train_state][self.slow_train_action]
            
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0

            new_value = old_value + self.alpha * (reward + self.gamma * max_next_q - old_value)
            self.q_table[self.slow_train_state][self.slow_train_action] = new_value
            
            self.slow_train_state = next_state
            self.slow_train_action = self.choose_action(self.slow_train_state)
            self.slow_train_path.append(next_state)
            self.slow_train_step_count += 1
        else:
            done = True

        if done:
            self.slow_train_episode_active = False
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.training_episode_count += 1

        return False, self.slow_train_path

    def extract_policy(self):
        """
        Extracts the greedy policy from the learned Q-table.
        """
        for state, actions in self.q_table.items():
            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                self.policy[state] = None
                continue
            
            valid_q_values = {action: actions.get(action, -np.inf) for action in valid_actions}
            
            if valid_q_values:
                self.policy[state] = max(valid_q_values, key=valid_q_values.get)
        self.is_trained = True

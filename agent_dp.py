import numpy as np
import random
from base_classes import BaseAgent

class DynamicProgrammingAgent(BaseAgent):
    """
    An agent that uses dynamic programming (policy iteration) to find the optimal policy.
    """
    def __init__(self, env, settings):
        super().__init__(env, settings)
        self.name = "Dynamic Programming"
        self.training_type = "iterative"
        self.gamma = float(settings.get('Discount Factor', 0.9))
        self.theta = float(settings.get('Theta', 1e-6)) # Read theta from settings
        self.reset()

    @staticmethod
    def get_editor_options():
        return {
            "Discount Factor": {"type": "input", "default": "0.9", "input_type": "float"},
            "Theta": {"type": "input", "default": "0.000001", "input_type": "float"},
        }

    def reset(self):
        """
        Resets the agent's value function and policy.
        The policy is now initialized with a random valid action for each state.
        """
        self.value_function = np.zeros((self.env.size, self.env.size, 2, 2))
        
        self.policy = np.full((self.env.size, self.env.size, 2, 2), None, dtype=object)
        for r in range(self.env.size):
            for c in range(self.env.size):
                for has_bag in range(2):
                    for has_rope in range(2):
                        state = (r, c, has_bag, has_rope)
                        valid_actions = self.env.get_valid_actions(state)
                        if valid_actions:
                            self.policy[state] = random.choice(valid_actions)
        
        self.is_trained = False

    def train_step(self):
        """
        Performs a single full iteration of Policy Iteration.
        1. Evaluates the current policy until the value function converges.
        2. Improves the policy based on the new value function.
        Returns whether the policy is stable and the final delta from evaluation.
        """
        # --- 1. Policy Evaluation ---
        # This loop runs until the value function for the current policy is stable.
        eval_delta = 0
        while True:
            eval_delta = 0
            # Create a copy to calculate new values based on the values from the previous sweep
            v_copy = np.copy(self.value_function)
            for state in self.env.state_space:
                v = self.value_function[state]
                action = self.policy[state]
                if action is None:
                    continue
                
                transitions = self.env.get_transition_model(state, action)
                # Calculate the new value using the values from the *previous* sweep (v_copy)
                new_v = sum(prob * (reward + self.gamma * v_copy[next_state]) for prob, next_state, reward in transitions)
                
                self.value_function[state] = new_v
                eval_delta = max(eval_delta, abs(v - new_v))
            
            # Check for convergence
            if eval_delta < self.theta:
                break

        # --- 2. Policy Improvement ---
        policy_stable = True
        for state in self.env.state_space:
            old_action = self.policy[state]
            if old_action is None:
                continue

            action_values = {}
            for action in self.env.get_valid_actions(state):
                transitions = self.env.get_transition_model(state, action)
                action_values[action] = sum(prob * (reward + self.gamma * self.value_function[next_state]) for prob, next_state, reward in transitions)
            
            if action_values:
                best_action = max(action_values, key=action_values.get)
                self.policy[state] = best_action
                if old_action != best_action:
                    policy_stable = False

        self.is_trained = policy_stable
        # The delta returned here is the final, small delta from the evaluation's convergence.
        # The meaningful change is whether the policy became stable.
        return policy_stable, eval_delta

    def extract_policy(self):
        pass

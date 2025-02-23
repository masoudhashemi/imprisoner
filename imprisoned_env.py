import random

import gymnasium as gym
import yaml
from gymnasium import spaces


class ImprisonedEnv(gym.Env):
    """A custom OpenAI Gym environment for the text-based game 'Imprisoned'."""

    def __init__(self, config_path="imprisoned.yaml"):
        super(ImprisonedEnv, self).__init__()

        # Load the YAML file
        with open(config_path, "r") as file:
            self.game_data = yaml.safe_load(file)

        self.states = self.game_data.get("states", {})
        self.starting_states = self.game_data.get("starting_states", [])

        # Verify starting states exist in the states dictionary
        self.starting_states = [s for s in self.starting_states if s in self.states]

        if not self.starting_states:
            raise ValueError("No valid starting states found in the YAML!")

        self.current_state = random.choice(self.starting_states)
        self.terminal = False
        self.inventory = set()  # Track collected items

        self.action_space = spaces.Discrete(len(self.get_available_actions()))
        self.observation_space = spaces.Discrete(len(self.states))

    def get_available_actions(self):
        """Returns available actions, considering inventory-based conditions."""
        if self.current_state not in self.states:
            print(f"⚠️ WARNING: State '{self.current_state}' not found in YAML!")
            return []

        actions = self.states[self.current_state].get("actions", {})
        available_actions = []

        for action, details in actions.items():
            conditions = details.get("conditions", {})
            required_item = conditions.get("requires")

            if not required_item or required_item in self.inventory:
                available_actions.append(action)

        return available_actions

    def get_state_description(self):
        """Returns the description of the current state."""
        return self.states.get(self.current_state, {}).get("description", "Unknown state.")

    def step(self, action_index):
        """Takes an action and transitions to a new state."""
        actions = self.get_available_actions()
        if not actions:
            print(f"⚠️ WARNING: No actions available in state '{self.current_state}'")
            return self.current_state, -1, True, {}

        if action_index >= len(actions):
            print("⚠️ WARNING: Invalid action index selected.")
            return self.current_state, -1, True, {}

        chosen_action = actions[action_index]
        action_data = self.states[self.current_state]["actions"][chosen_action]

        # Handle inventory item acquisition
        if "grants" in action_data:
            self.inventory.add(action_data["grants"])

        # Determine the next state
        if "probabilities" in action_data:
            next_states = list(action_data["probabilities"].keys())
            probabilities = list(action_data["probabilities"].values())
            new_state = random.choices(next_states, probabilities)[0]
        else:
            new_state = action_data.get("next_state", self.current_state)

        # Ensure the new state exists
        if new_state not in self.states:
            print(f"⚠️ WARNING: Next state '{new_state}' not found! Returning to a safe fallback.")
            new_state = random.choice(self.starting_states)

        # Update current state
        self.current_state = new_state
        self.terminal = self.states[self.current_state].get("terminal", False)

        # Check if we're in a state with no actions
        if not self.get_available_actions() and not self.terminal:
            print(f"⚠️ WARNING: State '{self.current_state}' has no actions!")
            # Return to a starting state if we're stuck
            self.current_state = random.choice(self.starting_states)

        # Only give reward of 1 for specifically reaching escape_success state
        reward = (
            1
            if self.current_state == "escape_success"
            else (-1 if self.terminal else 0)  # Penalty for other terminal states (death/capture)
        )

        return self.current_state, reward, self.terminal, {}

    def reset(self):
        """Resets the environment, clearing inventory."""
        self.current_state = random.choice(self.starting_states)
        while self.current_state not in self.states:
            print(f"⚠️ WARNING: {self.current_state} is not in the YAML states!")
            self.current_state = random.choice(self.starting_states)

        self.terminal = False
        self.inventory.clear()
        return self.current_state

    def render(self):
        """Displays the current state description and inventory."""
        print(f"\n🔹 {self.get_state_description()}")
        print(f"📜 Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}")

    def close(self):
        pass

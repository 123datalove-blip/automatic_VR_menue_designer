# rl_agent.py
import math
import random
import numpy as np
import pickle
import os
# Import necessary items from config explicitly
from config import (RL_PARAMS, Q_TABLE_PATH, SHAPE_OPTIONS, LAYOUT_OPTIONS,
                    MIN_EPSILON, EPSILON_DECAY, ADAPTABLE_PARAMS, INITIAL_PARAMS)
from user_model import UserModelGPR # Import the GPR model

class PersonalizationAgent:
    def __init__(self, initial_params, n_gpr_features=8): # Pass num features for GPR
        # RL Params
        self.alpha = RL_PARAMS['alpha']
        self.gamma = RL_PARAMS['gamma']
        self.epsilon = RL_PARAMS['initial_epsilon']
        self.actions = RL_PARAMS['actions']
        self.action_effects = RL_PARAMS['action_effects']
        self.param_bounds = RL_PARAMS['param_bounds']
        self.gpr_exploration_factor = initial_params.get('gpr_exploration_factor', 0.1)

        # State Discretization Bins
        self.size_bins = RL_PARAMS['size_bins']
        self.radius_bins = RL_PARAMS['radius_bins']
        self.avg_time_bins = RL_PARAMS['avg_time_bins']
        self.avg_error_bins = RL_PARAMS['avg_error_bins']
        self.path_efficiency_bins = RL_PARAMS['path_efficiency_bins']
        self.jerk_metric_bins = RL_PARAMS['jerk_metric_bins']
        self.gpr_uncertainty_bins = RL_PARAMS['gpr_uncertainty_bins']

        # Q-table & Params
        self.q_table = {}
        # Start with *all* adaptable params defaults, then merge initial settings passed in
        # Crucially, store the fixed num_items passed during initialization
        self.fixed_num_items = initial_params.get('num_items', 8) # Store the intended number
        self.current_params = {**ADAPTABLE_PARAMS} # Start with adaptable defaults ONLY
        # Merge initial settings for adaptable params, ensuring num_items is NOT overwritten here
        for key in ADAPTABLE_PARAMS.keys():
            if key in initial_params:
                self.current_params[key] = initial_params[key]
        # Explicitly set num_items from the fixed value passed in
        self.current_params['num_items'] = self.fixed_num_items
        # Add other fixed params needed for decisions (like interaction mode)
        if 'interaction_mode' in initial_params:
            self.current_params['interaction_mode'] = initial_params['interaction_mode']

        print(f"DEBUG (Agent Init): Initializing agent with fixed_num_items = {self.fixed_num_items}")
        print(f"DEBUG (Agent Init): Agent current_params = {self.current_params}")

        self.load_q_table() # Try loading saved table (will handle num_items correctly now)

        # User Model (GPR)
        self.user_model = UserModelGPR(n_features=n_gpr_features) # Match features used in _prepare_features

    # --- get_state, choose_action, update methods remain the same ---
    def get_state(self, avg_performance, trajectory_features, gpr_prediction):
        """ Discretizes combined state: params, avg perf, trajectory, GPR uncertainty. """
        # --- Extract values ---
        size = self.current_params['item_size']
        radius = self.current_params['layout_radius']
        shape_str = self.current_params.get('shape', SHAPE_OPTIONS[0])
        layout_str = self.current_params.get('layout_algorithm', LAYOUT_OPTIONS[0])
        avg_time = avg_performance.get('avg_time', 2.0)
        avg_error = avg_performance.get('avg_error', 50.0)
        path_eff = trajectory_features.get('path_efficiency', 0.5)
        jerk = trajectory_features.get('jerk_metric', 1e6)
        gpr_uncertainty = gpr_prediction.get('uncertainty', 1.0)

        # --- Discretize ---
        size_bin = np.digitize(size, self.size_bins)
        radius_bin = np.digitize(radius, self.radius_bins)
        avg_time_bin = np.digitize(avg_time, self.avg_time_bins)
        avg_error_bin = np.digitize(avg_error, self.avg_error_bins)
        path_eff_bin = np.digitize(path_eff, self.path_efficiency_bins)
        jerk_bin = np.digitize(jerk, self.jerk_metric_bins)
        gpr_unc_bin = np.digitize(gpr_uncertainty, self.gpr_uncertainty_bins)
        try: shape_idx = SHAPE_OPTIONS.index(shape_str)
        except ValueError: shape_idx = 0
        try: layout_idx = LAYOUT_OPTIONS.index(layout_str)
        except ValueError: layout_idx = 0

        # --- Create State Tuple (ORDER MATTERS!) ---
        state = (
            size_bin, radius_bin, shape_idx, layout_idx,
            avg_time_bin, avg_error_bin, path_eff_bin, jerk_bin, gpr_unc_bin
        )
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return state

    def choose_action(self, state, gpr_prediction):
        """ Epsilon-greedy action selection, biased by GPR uncertainty. """
        if random.random() < self.epsilon:
            action_index = random.randrange(len(self.actions))
        else:
            q_values = self.q_table.get(state, np.zeros(len(self.actions)))
            uncertainty_bonus = self.gpr_exploration_factor * gpr_prediction.get('uncertainty', 0.0)
            modified_q = q_values + uncertainty_bonus
            max_q = np.max(modified_q)
            best_actions = [i for i, q in enumerate(modified_q) if np.isclose(q, max_q)]
            action_index = random.choice(best_actions)
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        return action_index

    def update(self, state, action_index, scalar_reward, next_state):
        """ Updates the Q-table using scalarized reward. """
        if state not in self.q_table: self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table: self.q_table[next_state] = np.zeros(len(self.actions))
        old_value = self.q_table[state][action_index]
        next_max_q = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (scalar_reward + self.gamma * next_max_q - old_value)
        self.q_table[state][action_index] = new_value

    def apply_action_and_get_new_params(self, action_index):
        """ Applies action, updates self.current_params, returns new params dict. """
        action_name = self.actions[action_index]
        new_params = self.current_params.copy() # Operate on a copy

        # --- IMPORTANT: Store num_items before applying effects ---
        original_num_items = new_params.get('num_items', self.fixed_num_items)
        # ---

        if action_name == 'next_shape':
            current_idx = SHAPE_OPTIONS.index(new_params.get('shape', SHAPE_OPTIONS[0]))
            next_idx = (current_idx + 1) % len(SHAPE_OPTIONS)
            new_params['shape'] = SHAPE_OPTIONS[next_idx]
        elif action_name == 'next_layout':
            current_idx = LAYOUT_OPTIONS.index(new_params.get('layout_algorithm', LAYOUT_OPTIONS[0]))
            next_idx = (current_idx + 1) % len(LAYOUT_OPTIONS)
            new_params['layout_algorithm'] = LAYOUT_OPTIONS[next_idx]
        elif action_name != 'no_change':
            effects = self.action_effects.get(action_name, {})
            for param, change in effects.items():
                if param in new_params:
                    if 'angle' in param:
                         new_value = new_params[param] + change
                         min_b, max_b = self.param_bounds.get(param, (-math.pi, math.pi))
                         if param == 'arc_start_angle': new_value = min(new_value, new_params.get('arc_end_angle', max_b) - 0.1)
                         elif param == 'arc_end_angle': new_value = max(new_value, new_params.get('arc_start_angle', min_b) + 0.1)
                         new_params[param] = np.clip(new_value, min_b, max_b)
                    else:
                        new_value = new_params[param] + change
                        min_b, max_b = self.param_bounds.get(param, (-np.inf, np.inf))
                        new_params[param] = np.clip(new_value, min_b, max_b)

        # --- IMPORTANT: Ensure num_items was NOT changed by effects ---
        if 'num_items' in new_params and new_params['num_items'] != original_num_items:
             print(f"CRITICAL WARNING (apply_action): num_items changed unexpectedly from {original_num_items} to {new_params['num_items']}! Reverting.")
             new_params['num_items'] = original_num_items
        elif 'num_items' not in new_params: # Ensure it exists
             new_params['num_items'] = original_num_items
        # ---

        # Update the agent's internal state
        self.current_params = new_params
        return self.current_params

    # --- Persistence ---
    def save_q_table(self, filename=Q_TABLE_PATH):
        try:
            # --- IMPORTANT: Ensure saved params have correct fixed num_items ---
            params_to_save = self.current_params.copy()
            params_to_save['num_items'] = self.fixed_num_items # Save the fixed value
            # ---
            save_data = {'q_table': self.q_table, 'epsilon': self.epsilon, 'current_params': params_to_save}
            with open(filename, 'wb') as f: pickle.dump(save_data, f)
            print(f"Q-table saved to {filename}")
            if self.user_model: self.user_model.save_model()
        except Exception as e: print(f"Error saving Q-table: {e}")

    def load_q_table(self, filename=Q_TABLE_PATH):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f: load_data = pickle.load(f)
                self.q_table = load_data.get('q_table', {})
                self.epsilon = load_data.get('epsilon', RL_PARAMS['initial_epsilon'])
                loaded_params = load_data.get('current_params')
                if loaded_params:
                     print(f"DEBUG (load_q_table): Loaded params from file: {loaded_params}")
                     # --- IMPORTANT: Merge loaded params BUT keep fixed num_items ---
                     # Start with adaptable defaults
                     merged_params = {**ADAPTABLE_PARAMS}
                     # Update with loaded adaptable params ONLY
                     for key in ADAPTABLE_PARAMS.keys():
                          if key in loaded_params:
                               merged_params[key] = loaded_params[key]
                     # Set the fixed num_items from the agent's init value
                     merged_params['num_items'] = self.fixed_num_items
                     # Add interaction mode from current agent state (or default if needed)
                     merged_params['interaction_mode'] = self.current_params.get('interaction_mode', INITIAL_PARAMS['interaction_mode'])
                     # ---
                     self.current_params = merged_params
                     print(f"DEBUG (load_q_table): Merged current_params: {self.current_params}")

                print(f"Q-table loaded from {filename}")
            except Exception as e:
                print(f"Error loading Q-table: {e}. Resetting.")
                self.q_table = {}
                # Reset current_params to defaults if loading fails?
                self.current_params = {**ADAPTABLE_PARAMS, 'num_items': self.fixed_num_items, 'interaction_mode': INITIAL_PARAMS['interaction_mode']}

        else:
            print("No saved Q-table found. Starting fresh.")

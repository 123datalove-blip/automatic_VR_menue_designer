# config.py
import math
import os
import numpy as np

# --- Persistence ---
Q_TABLE_FILENAME = "vrp_adv_q_table.pkl"
GPR_MODEL_FILENAME = "vrp_gpr_model.pkl"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Q_TABLE_PATH = os.path.join(SCRIPT_DIR, Q_TABLE_FILENAME)
GPR_MODEL_PATH = os.path.join(SCRIPT_DIR, GPR_MODEL_FILENAME)


# --- GUI Settings ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 750
CANVAS_BG = "white"
CENTER_X = WINDOW_WIDTH // 2
DRAWING_AREA_HEIGHT = WINDOW_HEIGHT - 200 # Canvas height = 550
CENTER_Y = DRAWING_AREA_HEIGHT // 2 # Center Y = 275

# --- Define Max Item Size for Bound Calculation ---
# Get max possible item size from param_bounds if defined, else use a default
MAX_ITEM_SIZE = 60 # Default max size if not found below

# --- RL Agent Settings ---
# (Define RL_PARAMS *before* using its bounds)
INITIAL_EPSILON = 0.4
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05
HISTORY_LENGTH = 5

RL_PARAMS = {
    'alpha': 0.1,
    'gamma': 0.9,
    'initial_epsilon': INITIAL_EPSILON,
    'actions': [
        'increase_size', 'decrease_size', 'increase_radius', 'decrease_radius',
        'next_shape', 'next_layout',
        'increase_k_repel', 'decrease_k_repel', 'increase_k_attract', 'decrease_k_attract',
        'increase_arc_angle', 'decrease_arc_angle', 'shift_arc_center_h', 'shift_arc_center_v',
        'no_change'
    ],
    # Bins
    'size_bins': [15, 25, 35, 45, 55],
    'radius_bins': [60, 100, 140, 180, 220], # Adjusted based on MAX_SAFE_RADIUS below
    'avg_time_bins': [0.5, 1.0, 1.5, 2.0, 2.5],
    'avg_error_bins': [5, 15, 30, 50, 70],
    'path_efficiency_bins': [0.5, 0.7, 0.8, 0.9],
    'jerk_metric_bins': [1e3, 1e4, 1e5, 1e6],
    'gpr_uncertainty_bins': [0.1, 0.5, 1.0],

    # Action Effects
    'action_effects': {
        'increase_size': {'item_size': 5}, 'decrease_size': {'item_size': -5},
        'increase_radius': {'layout_radius': 15}, 'decrease_radius': {'layout_radius': -15},
        'increase_k_repel': {'force_k_repel': 1000.0}, 'decrease_k_repel': {'force_k_repel': -1000.0},
        'increase_k_attract': {'force_k_attract': 0.05}, 'decrease_k_attract': {'force_k_attract': -0.05},
        'increase_arc_angle': {'arc_start_angle': -0.1, 'arc_end_angle': 0.1},
        'decrease_arc_angle': {'arc_start_angle': 0.1, 'arc_end_angle': -0.1},
        'shift_arc_center_h': {'arc_center_offset_x': 10}, 'shift_arc_center_v': {'arc_center_offset_y': 10},
        'no_change': {}
    },
    # Parameter bounds
    'param_bounds': {
        'item_size': (10, 60),
        # Max radius calculation will happen *after* this dict is defined
        'layout_radius': (60, 250), # Placeholder, will be overwritten below
        'force_k_repel': (1000.0, 20000.0), 'force_k_attract': (0.01, 0.5),
        'arc_center_offset_x': (-150, 150), 'arc_center_offset_y': (-100, 100),
        'arc_start_angle': (-math.pi + 0.1, -0.1), 'arc_end_angle': (0.1, math.pi - 0.1),
    }
}

# --- Now calculate MAX_SAFE_RADIUS using defined bounds ---
if 'item_size' in RL_PARAMS['param_bounds']:
    MAX_ITEM_SIZE = RL_PARAMS['param_bounds']['item_size'][1] # Get max size from bounds

# Calculate max radius considering max item size and canvas dimensions
# Subtract half max item size + a margin (e.g., 10px) from half the drawing height/width
MARGIN = 10
max_rad_y = DRAWING_AREA_HEIGHT // 2 - (MAX_ITEM_SIZE / 2) - MARGIN
max_rad_x = WINDOW_WIDTH // 2 - (MAX_ITEM_SIZE / 2) - MARGIN
MAX_SAFE_RADIUS = max(10, min(max_rad_x, max_rad_y)) # Use the smaller dimension, ensure > 0
print(f"INFO: Calculated MAX_SAFE_RADIUS = {MAX_SAFE_RADIUS}")

# --- Update the param_bounds for layout_radius ---
RL_PARAMS['param_bounds']['layout_radius'] = (60, MAX_SAFE_RADIUS)
# --- Update radius bins if needed based on new max ---
RL_PARAMS['radius_bins'] = np.linspace(60, MAX_SAFE_RADIUS, 5).tolist() # Example: 5 bins up to max
print(f"INFO: Updated layout_radius bounds to: {RL_PARAMS['param_bounds']['layout_radius']}")
print(f"INFO: Updated radius_bins to: {RL_PARAMS['radius_bins']}")


# --- Menu Item Options ---
DEFAULT_NUM_ITEMS = 8
SHAPE_OPTIONS = ['circle', 'square', 'ellipse', 'triangle', 'pentagon', 'hexagon']
LAYOUT_OPTIONS = ['circular', 'clustered_dominant_hand', 'force_directed', 'ergonomic_arc']
COLOR_OPTIONS = {"Blue": "lightblue", "Green": "lightgreen", "Gray": "lightgray", "Pink": "pink", "Orange": "orange", "Yellow": "lightyellow"}
FONT_OPTIONS = [("Arial", 8), ("Arial", 10), ("Arial", 12), ("Times", 10)]
INTERACTION_MODES = ["Simulated Two-Handed", "Simulated One-Handed"]

# Parameters that can be adapted by RL (num_items REMOVED)
ADAPTABLE_PARAMS = {
    'item_size': 30,
    'layout_radius': 150,
    'shape': 'circle',
    'layout_algorithm': 'circular',
    'force_k_repel': 5000.0, 'force_k_attract': 0.1, 'force_optimal_dist': 150.0,
    'arc_center_offset_x': 0, 'arc_center_offset_y': 50,
    'arc_start_angle': -math.pi * 0.6, 'arc_end_angle': math.pi * 0.6,
    'gpr_exploration_factor': 0.1,
    'ellipse_stretch_ratio': 1.5, 'polygon_sides': 0,
}

# Parameters usually set initially and fixed (num_items REMOVED)
FIXED_PARAMS = {
    'color': 'lightblue', 'font': ("Arial", 10), 'interaction_mode': INTERACTION_MODES[0],
}

# Default Initial Menu Parameters
INITIAL_PARAMS = {**ADAPTABLE_PARAMS, **FIXED_PARAMS}


# --- Performance Calculation ---
FITTS_A = 0.10; FITTS_B = 0.15
REWARD_WEIGHTS = {
    'speed': 1.0, 'accuracy': 1.5, 'fitts_bonus': 0.6,
    'path_efficiency': 0.4, 'jerk_penalty': 0.05,
    'distance_penalty_one_handed': 0.005
}

# --- Voronoi ---
VORONOI_BOUNDS_MARGIN = 50

# --- Trajectory Analysis ---
JERK_DT = 0.02

# --- Force Layout ---
FORCE_ITERATIONS = 50
FORCE_DAMPING = 0.7


# layout_engine.py
import numpy as np
import math
# Import necessary constants from config
from config import FORCE_ITERATIONS, FORCE_DAMPING, CENTER_X, CENTER_Y, WINDOW_WIDTH, WINDOW_HEIGHT

def _is_valid_coord(coord):
    """Checks if a coordinate is a finite number."""
    return isinstance(coord, (int, float)) and np.isfinite(coord)

# --- Accept num_items as argument ---
def calculate_force_directed_layout(num_items, current_positions, params):
    """
    Calculates item positions using a force-directed algorithm.

    Args:
        num_items (int): The fixed number of items to calculate positions for.
        current_positions: list of {'x': float, 'y': float, 'id': int} or None (used for initialization)
        params (dict): Dictionary containing force parameters and other adaptable settings.

    Returns:
        list of {'x': float, 'y': float, 'id': int}
    """
    if num_items <= 0: return [] # Handle case of zero items

    k_repel = params.get('force_k_repel', 5000.0)
    k_attract = params.get('force_k_attract', 0.1)
    optimal_dist = params.get('force_optimal_dist', params.get('layout_radius', 150))

    # Initialize positions
    if current_positions and len(current_positions) == num_items: # Use previous positions if valid
        positions = np.array([[p['x'], p['y']] for p in current_positions])
    else: # Initialize randomly if no valid previous positions
        print("DEBUG (force_directed): Initializing positions randomly.")
        positions = np.random.rand(num_items, 2) * 100 + np.array([CENTER_X - 50, CENTER_Y - 50])

    velocities = np.zeros_like(positions)

    # --- Force calculation loop (remains the same) ---
    for iter_count in range(FORCE_ITERATIONS):
        forces = np.zeros_like(positions)
        # Repulsive forces
        for i in range(num_items):
            for j in range(i + 1, num_items):
                delta = positions[i] - positions[j]
                distance_sq = max(np.sum(delta**2), 1.0)
                distance = np.sqrt(distance_sq)
                direction = delta / distance
                repulsive_force = (k_repel / distance_sq) * direction
                forces[i] += repulsive_force
                forces[j] -= repulsive_force
        # Attractive forces
        center = np.array([CENTER_X, CENTER_Y])
        for i in range(num_items):
            delta_to_center = positions[i] - center
            distance_to_center = max(np.linalg.norm(delta_to_center), 1.0)
            direction_to_center = delta_to_center / distance_to_center
            # Prevent log(0) or negative values if optimal_dist is very small/zero
            log_arg = max(1e-6, distance_to_center / max(1e-6, optimal_dist))
            attractive_force = k_attract * math.log(log_arg) * direction_to_center
            forces[i] -= attractive_force
        # Update velocities and positions
        velocities = (velocities + forces) * FORCE_DAMPING
        positions += velocities

        # Optional: Constraint nodes within canvas bounds
        # drawing_height = WINDOW_HEIGHT - 200
        # positions[:, 0] = np.clip(positions[:, 0], 50, WINDOW_WIDTH - 50)
        # positions[:, 1] = np.clip(positions[:, 1], 50, drawing_height - 50)

        # Optional: Check for NaN/Inf during iterations
        if not np.all(np.isfinite(positions)):
            print(f"ERROR (force_directed): Non-finite positions detected at iteration {iter_count+1}! Stopping.")
            # Return empty list or last valid positions? Returning empty for safety.
            return []

    # --- Validate final positions ---
    final_positions = []
    for i, pos in enumerate(positions):
        if _is_valid_coord(pos[0]) and _is_valid_coord(pos[1]):
            final_positions.append({'x': pos[0], 'y': pos[1], 'id': i})
        else:
            print(f"ERROR (force_directed): Invalid final position for item {i}! Pos={pos}. Skipping.")

    # --- Final check on count ---
    if len(final_positions) != num_items:
         print(f"ERROR (force_directed): Returned {len(final_positions)} positions, expected {num_items}!")

    return final_positions


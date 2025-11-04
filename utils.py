# utils.py
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
# Import necessary constants from config
from config import (CENTER_X, CENTER_Y, JERK_DT, SHAPE_OPTIONS, LAYOUT_OPTIONS,
                   VORONOI_BOUNDS_MARGIN as VBM, INTERACTION_MODES)


def calculate_distance(x1, y1, x2, y2):
    """Calculates Euclidean distance."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_midpoint(coords):
    """Gets the midpoint from Tkinter coords (x0, y0, x1, y1)."""
    if not coords or len(coords) < 4: return (0, 0)
    return ((coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2)

def get_regular_polygon_vertices(center_x, center_y, radius, sides, rotation=0):
    """Calculates vertices for a regular polygon."""
    if sides < 3: return []
    vertices = []
    angle_step = 2 * math.pi / sides
    for i in range(sides):
        angle = angle_step * i + rotation - math.pi / 2
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        vertices.extend([x, y])
    return vertices

def _is_valid_coord(coord):
    """Checks if a coordinate is a finite number."""
    return isinstance(coord, (int, float)) and np.isfinite(coord)

# --- Layout Calculation Functions (Take num_items as arg) ---

def calculate_circular_or_ellipse_positions(center_x, center_y, num_items, radius, params):
    """Calculates positions evenly spaced on a circle or ellipse."""
    positions = []
    if num_items <= 0: return positions
    angle_step = 2 * math.pi / num_items
    shape = params.get('shape', 'circle')
    stretch = params.get('ellipse_stretch_ratio', 1.5)
    start_offset = 0.0001

    # print(f"DEBUG (calculate_circular): num_items={num_items}, angle_step={angle_step:.4f}")

    for i in range(num_items):
        angle = start_offset + (angle_step * i)
        if shape == 'ellipse':
            a = radius * stretch; b = radius / stretch
            x = center_x + a * math.cos(angle); y = center_y + b * math.sin(angle)
        else:
            x = center_x + radius * math.cos(angle); y = center_y + radius * math.sin(angle)

        if _is_valid_coord(x) and _is_valid_coord(y):
            positions.append({'x': x, 'y': y, 'id': i})
        else:
            print(f"ERROR (calculate_circular): Invalid coords for item {i}! x={x}, y={y}. Skipping.")

    if len(positions) != num_items:
        print(f"ERROR (calculate_circular): Generated {len(positions)} positions, expected {num_items}!")

    return positions

def calculate_clustered_positions(center_x, center_y, num_items, radius, params):
    """ Calculates positions clustered in a specific arc (1H) or delegates to circular (2H). """
    positions = []
    if num_items <= 0: return positions

    interaction_mode = params.get('interaction_mode', "Simulated Two-Handed")

    if interaction_mode == INTERACTION_MODES[1]: # One-Handed
         cluster_center_angle = 0; cluster_spread_angle_one_handed = math.pi * 0.8
         angle_range = cluster_spread_angle_one_handed
         angle_step = angle_range / max(1, num_items - 1) if num_items > 1 else 0
         start_angle = cluster_center_angle - angle_range / 2
         # print(f"DEBUG (calculate_clustered - 1H): range={angle_range:.2f}, step={angle_step:.2f}, start={start_angle:.2f}")

         for i in range(num_items):
             angle = start_angle + angle_step * i if num_items > 1 else start_angle
             x = center_x + radius * math.cos(angle)
             y = center_y + radius * math.sin(angle)
             if _is_valid_coord(x) and _is_valid_coord(y):
                 positions.append({'x': x, 'y': y, 'id': i})
             else:
                 print(f"ERROR (calculate_clustered - 1H): Invalid coords for item {i}! x={x}, y={y}. Skipping.")

    else: # Two-Handed: Delegate to standard circular/ellipse function
         # print(f"DEBUG (calculate_clustered - 2H): Delegating to calculate_circular_or_ellipse_positions")
         positions = calculate_circular_or_ellipse_positions(center_x, center_y, num_items, radius, params)

    if len(positions) != num_items:
        print(f"ERROR (calculate_clustered): Generated {len(positions)} positions, expected {num_items}!")

    return positions


def calculate_arc_positions(center_x, center_y, num_items, radius, params):
    """ Calculates positions along a potentially offset arc segment. """
    positions = []
    if num_items <= 0: return positions

    start_angle = params.get('arc_start_angle', -math.pi/2)
    end_angle = params.get('arc_end_angle', math.pi/2)
    offset_x = params.get('arc_center_offset_x', 0)
    offset_y = params.get('arc_center_offset_y', 0)
    arc_center_x = center_x + offset_x
    arc_center_y = center_y + offset_y

    angle_range = end_angle - start_angle
    if angle_range <= 0:
        print(f"Warning (calculate_arc): Start angle {start_angle:.2f} >= end angle {end_angle:.2f}. Using default range.")
        start_angle = -math.pi/2; end_angle = math.pi/2; angle_range = math.pi

    angle_step = angle_range / max(1, num_items - 1) if num_items > 1 else 0

    for i in range(num_items):
        angle = start_angle + angle_step * i
        x = arc_center_x + radius * math.cos(angle)
        y = arc_center_y + radius * math.sin(angle)

        if _is_valid_coord(x) and _is_valid_coord(y):
            positions.append({'x': x, 'y': y, 'id': i})
        else:
            print(f"ERROR (calculate_arc): Invalid coords for item {i}! x={x}, y={y}. Skipping.")
    return positions


# --- Voronoi Helper ---
def get_voronoi_cells(points, canvas_width, canvas_height):
    # (Keep Voronoi logic as is, it depends on the points passed to it)
    if len(points) < 4: return {}, None
    dummy_dist = max(canvas_width, canvas_height) * 5; points_array = np.array(points)
    canvas_center_x = canvas_width / 2; canvas_center_y = canvas_height / 2
    dummy_points = np.array([[canvas_center_x - dummy_dist, canvas_center_y - dummy_dist],[canvas_center_x + dummy_dist, canvas_center_y - dummy_dist],[canvas_center_x + dummy_dist, canvas_center_y + dummy_dist],[canvas_center_x - dummy_dist, canvas_center_y + dummy_dist]])
    unique_original_points, _ = np.unique(points_array, axis=0, return_index=True)
    if len(unique_original_points) < len(points_array): print("Warning: Duplicate points found in input for Voronoi."); points_array = unique_original_points
    all_points = np.vstack((points_array, dummy_points)); unique_all_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    if len(unique_all_points) < 4: print("Warning: Less than 4 unique points after adding dummies."); return {}, None
    try: vor = Voronoi(unique_all_points)
    except Exception as e: print(f"Error creating Voronoi diagram: {e}"); return {}, None
    min_x, max_x = -VBM, canvas_width + VBM; min_y, max_y = -VBM, canvas_height + VBM
    regions = {}; original_indices_in_unique = unique_indices[unique_indices < len(points_array)]
    for i, unique_idx in enumerate(original_indices_in_unique):
        point_region_idx = vor.point_region[unique_idx]; region_vertex_indices = vor.regions[point_region_idx]
        if region_vertex_indices and -1 not in region_vertex_indices:
            vertices = vor.vertices[region_vertex_indices]
            if np.all((vertices[:, 0] >= min_x) & (vertices[:, 0] <= max_x) & (vertices[:, 1] >= min_y) & (vertices[:, 1] <= max_y)):
                 original_index = i; regions[original_index] = vertices.tolist()
            else: regions[i] = []
        else: regions[i] = []
    return regions, vor


def point_in_polygon(x, y, polygon_vertices):
    # (Keep point_in_polygon logic as is)
    if not polygon_vertices or len(polygon_vertices) < 3: return False
    num_vertices = len(polygon_vertices); inside = False; px, py = x, y
    p1x, p1y = polygon_vertices[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon_vertices[i % num_vertices]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y: xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# --- Trajectory Helpers ---
def calculate_trajectory_features(path_points):
    # (Keep trajectory calculation logic as is)
    features = {'path_length': 0, 'directness': 1.0, 'jerk_metric': 0}
    if not path_points or len(path_points) < 3: return features
    points = np.array(path_points); n_points = len(points)
    path_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    features['path_length'] = path_length
    start_point = points[0]; end_point = points[-1]
    straight_dist = calculate_distance(start_point[0], start_point[1], end_point[0], end_point[1])
    if straight_dist > 1e-6: features['directness'] = straight_dist / path_length if path_length > 1e-6 else 1.0
    else: features['directness'] = 1.0 if path_length < 1e-6 else 0.0
    dt = JERK_DT
    if n_points >= 4:
        velocities = np.diff(points, axis=0) / dt
        accelerations = np.diff(velocities, axis=0) / dt
        jerks = np.diff(accelerations, axis=0) / dt
        jerk_metric = np.sum(np.sum(jerks**2, axis=1))
        features['jerk_metric'] = jerk_metric
    return features

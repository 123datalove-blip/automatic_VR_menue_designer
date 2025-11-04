# performance_monitor.py
import time
import math
import collections
import numpy as np # Add numpy
from utils import calculate_distance, get_midpoint, calculate_trajectory_features # Add trajectory util
from config import (REWARD_WEIGHTS, INTERACTION_MODES, FITTS_A, FITTS_B, HISTORY_LENGTH,
                   JERK_DT) # Add JERK_DT

class PerformanceMonitor:
    def __init__(self):
        self.start_time = 0
        self.last_performance = {}
        self.performance_history = collections.deque(maxlen=HISTORY_LENGTH)
        self.current_trajectory = [] # Store path points
        self.is_recording_trajectory = False

    def start(self):
        self.start_time = time.time()
        self.current_trajectory = []
        self.is_recording_trajectory = True # Start recording path

    def record_motion(self, event):
        """Records mouse position during movement if active."""
        if self.is_recording_trajectory:
            self.current_trajectory.append((event.x, event.y))

    def stop(self, click_event, target_canvas_id, canvas):
        self.is_recording_trajectory = False # Stop recording path
        if self.start_time == 0: return None

        actual_time = time.time() - self.start_time
        # Add click point to trajectory end
        if self.current_trajectory: # Ensure recording happened
             self.current_trajectory.append((click_event.x, click_event.y))

        predicted_time = 0; fitts_id = 0; click_error = 0
        distance_to_target = 0; target_width = 1; target_coords = None

        try:
            target_coords = canvas.coords(target_canvas_id)
            if not target_coords: raise ValueError("Target item not found.")
            target_x, target_y = get_midpoint(target_coords)
            target_width = max(1, target_coords[2] - target_coords[0])

            click_x, click_y = click_event.x, click_event.y
            click_error = calculate_distance(click_x, click_y, target_x, target_y)

            # Fitts
            start_x, start_y = self.current_trajectory[0] if self.current_trajectory else (canvas.winfo_width()//2, canvas.winfo_height()//2)
            distance_to_target = calculate_distance(start_x, start_y, target_x, target_y)
            if target_width > 0 and distance_to_target > 0:
                 try:
                     fitts_id = math.log2((distance_to_target / target_width) + 1)
                     predicted_time = FITTS_A + FITTS_B * fitts_id
                 except ValueError: fitts_id = predicted_time = 0

            # Trajectory Features
            trajectory_features = calculate_trajectory_features(self.current_trajectory)
            # Path efficiency is inverse of directness measure from utils
            path_efficiency = trajectory_features.get('directness', 0.0)

            current_performance = {
                'time_taken': actual_time, 'click_error': click_error,
                'distance_to_target': distance_to_target, 'target_width': target_width,
                'fitts_id': fitts_id, 'predicted_time': predicted_time,
                'path_length': trajectory_features['path_length'],
                'path_efficiency': path_efficiency, # Higher is better (more direct)
                'jerk_metric': trajectory_features['jerk_metric'] # Lower is better
            }
            self.performance_history.append(current_performance) # Store full dict
            self.last_performance = current_performance
            self.start_time = 0
            return self.last_performance

        except Exception as e:
            print(f"Error getting performance: {e}")
            self.is_recording_trajectory = False; self.start_time = 0
            return None

    def get_average_performance(self):
        if not self.performance_history:
            # Return defaults reflecting poor performance initially
            return {'avg_time': 2.0, 'avg_error': 50.0, 'avg_path_efficiency': 0.5, 'avg_jerk': 1e6}

        count = len(self.performance_history)
        avg_time = sum(p['time_taken'] for p in self.performance_history) / count
        avg_error = sum(p['click_error'] for p in self.performance_history) / count
        avg_path_eff = sum(p['path_efficiency'] for p in self.performance_history) / count
        avg_jerk = sum(p['jerk_metric'] for p in self.performance_history) / count

        return {
            'avg_time': avg_time, 'avg_error': avg_error,
            'avg_path_efficiency': avg_path_eff, 'avg_jerk': avg_jerk
        }

    def calculate_reward_vector(self, performance_metrics, interaction_mode):
        """ Calculates a VECTOR of rewards based on multiple objectives. """
        rewards = {
            'speed': 0, 'accuracy': 0, 'fitts_bonus': 0,
            'path_efficiency': 0, 'jerk_penalty': 0,
            'distance_penalty': 0
        }
        if not performance_metrics: return rewards

        actual_time = performance_metrics['time_taken']
        click_error = performance_metrics['click_error']
        predicted_time = performance_metrics['predicted_time']
        path_efficiency = performance_metrics['path_efficiency'] # 0 to 1 (higher better)
        jerk = performance_metrics['jerk_metric'] # Lower better

        # Speed (higher is better)
        rewards['speed'] = 1.0 / (1 + actual_time)
        # Accuracy (higher is better)
        rewards['accuracy'] = 1.0 / (1 + click_error)
        # Fitts Bonus (higher is better)
        fitts_deviation = predicted_time - actual_time
        rewards['fitts_bonus'] = max(0, fitts_deviation)
        # Path Efficiency (higher is better, scale 0-1)
        rewards['path_efficiency'] = path_efficiency
        # Jerk Penalty (lower jerk -> lower penalty -> higher reward component)
        # Need scaling - very large values possible. Use log or clip? Let's use inverse relation.
        rewards['jerk_penalty'] = 1.0 / (1 + jerk * 1e-6) # Heavily scaled inverse

        # Distance Penalty (One-Handed)
        if interaction_mode == INTERACTION_MODES[1]:
            rewards['distance_penalty'] = -REWARD_WEIGHTS['distance_penalty_one_handed'] * performance_metrics['distance_to_target']

        return rewards

    def scalarize_reward(self, reward_vector):
        """ Combines reward vector into a single scalar using weights from config. """
        scalar_reward = 0
        weights = REWARD_WEIGHTS # Use weights from config
        for key, value in reward_vector.items():
            scalar_reward += weights.get(key, 0) * value # Use weight if exists, else 0

        # Add distance penalty directly as it's already scaled/negative
        scalar_reward += reward_vector.get('distance_penalty', 0)

        return max(0, scalar_reward) # Ensure overall reward isn't negative (optional)
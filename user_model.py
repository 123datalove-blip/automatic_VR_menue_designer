# user_model.py
import numpy as np
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from config import (GPR_MODEL_PATH, SHAPE_OPTIONS, LAYOUT_OPTIONS)

class UserModelGPR:
    def __init__(self, n_features, random_state=42):
        # Define kernel: RBF for non-linearity, ConstantKernel for scaling, WhiteKernel for noise
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2)) \
                      + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

        self.gpr_time = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, random_state=random_state, alpha=1e-5)
        self.gpr_error = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, random_state=random_state, alpha=1e-5) # Separate model for error

        self.scaler = StandardScaler() # Scale features for better GPR performance
        self.features = []
        self.times = []
        self.errors = []
        self.n_features = n_features
        self.is_fitted = False
        self.load_model()

    def _prepare_features(self, params, perf_history, trajectory_features):
        """ Extracts features from current state for GPR input. """
        # EXAMPLE features - This needs careful selection based on what influences performance
        feature_vec = [
            params.get('item_size', 30),
            params.get('layout_radius', 150),
            SHAPE_OPTIONS.index(params.get('shape', 'circle')),
            LAYOUT_OPTIONS.index(params.get('layout_algorithm', 'circular')),
            perf_history.get('avg_time', 1.0),
            perf_history.get('avg_error', 20.0),
            trajectory_features.get('path_efficiency', 0.8),
            trajectory_features.get('jerk_metric', 1e4),
        ]
        # Ensure correct number of features - pad if necessary (though should match n_features)
        feature_vec = feature_vec[:self.n_features]
        if len(feature_vec) < self.n_features:
             feature_vec.extend([0] * (self.n_features - len(feature_vec)))

        return np.array(feature_vec).reshape(1, -1)

    def update(self, params, performance_metrics, avg_performance, trajectory_features):
        """ Adds new data point and refits the GPR model periodically. """
        if not performance_metrics: return

        new_feature_vec = self._prepare_features(params, avg_performance, trajectory_features)

        self.features.append(new_feature_vec.flatten())
        self.times.append(performance_metrics['time_taken'])
        self.errors.append(performance_metrics['click_error'])

        # Refit occasionally (e.g., every 5-10 updates) - can be slow
        if len(self.features) > self.n_features and len(self.features) % 10 == 0:
            print("Refitting GPR Model...")
            X = np.array(self.features)
            y_time = np.array(self.times)
            y_error = np.array(self.errors)

            # Scale features
            X_scaled = self.scaler.fit_transform(X) # Fit scaler only on collected data

            try:
                 self.gpr_time.fit(X_scaled, y_time)
                 self.gpr_error.fit(X_scaled, y_error)
                 self.is_fitted = True
                 print("GPR Model Refit Complete.")
            except Exception as e:
                 print(f"GPR Fit Error: {e}")
                 self.is_fitted = False # Reset flag if fit fails


    def predict(self, params, avg_performance, trajectory_features):
        """ Predicts performance (time, error) and uncertainty for given features. """
        if not self.is_fitted or not self.features:
            # Return defaults or basic Fitts if model not ready
            return {'pred_time': 1.0, 'pred_error': 20.0, 'uncertainty': 1.0} # High uncertainty

        feature_vec = self._prepare_features(params, avg_performance, trajectory_features)
        feature_vec_scaled = self.scaler.transform(feature_vec) # Use fitted scaler

        try:
             pred_time, std_time = self.gpr_time.predict(feature_vec_scaled, return_std=True)
             pred_error, std_error = self.gpr_error.predict(feature_vec_scaled, return_std=True)

             # Combine uncertainties (e.g., average or max)
             uncertainty = max(0.01, (std_time[0] + std_error[0]) / 2.0) # Avoid zero uncertainty

             return {
                 'pred_time': max(0.1, pred_time[0]), # Ensure positive time
                 'pred_error': max(0.0, pred_error[0]), # Ensure non-negative error
                 'uncertainty': uncertainty
             }
        except Exception as e:
            print(f"GPR Predict Error: {e}")
            return {'pred_time': 1.0, 'pred_error': 20.0, 'uncertainty': 1.0}


    def save_model(self, filename=GPR_MODEL_PATH):
        """ Saves the GPR models and scaler. """
        if not self.is_fitted: return
        save_data = {
            'gpr_time': self.gpr_time,
            'gpr_error': self.gpr_error,
            'scaler': self.scaler,
            'features': self.features, # Save data used for fitting
            'times': self.times,
            'errors': self.errors
        }
        try:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"GPR Model saved to {filename}")
        except Exception as e:
            print(f"Error saving GPR Model: {e}")

    def load_model(self, filename=GPR_MODEL_PATH):
        """ Loads GPR models and scaler. """
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    load_data = pickle.load(f)
                self.gpr_time = load_data['gpr_time']
                self.gpr_error = load_data['gpr_error']
                self.scaler = load_data['scaler']
                self.features = load_data.get('features', []) # Load past data
                self.times = load_data.get('times', [])
                self.errors = load_data.get('errors', [])
                self.n_features = self.gpr_time.kernel_.get_params()['k2__k2__length_scale'].shape[0]
                self.is_fitted = True # Assume loaded model is fitted
                print(f"GPR Model loaded from {filename} (Features: {self.n_features}, Data Points: {len(self.features)})")
            except Exception as e:
                print(f"Error loading GPR Model: {e}. Starting fresh.")
                self.is_fitted = False
        else:
             print("No saved GPR model found.")
             self.is_fitted = False
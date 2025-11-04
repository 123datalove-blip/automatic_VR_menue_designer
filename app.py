# app.py
import tkinter as tk
from tkinter import simpledialog, messagebox, font, Frame, Label, Checkbutton, BooleanVar
import random
import numpy as np
import time
import os
from datetime import datetime 
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Optional for embedding
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from utils import point_in_polygon, get_midpoint
# Import necessary items from config explicitly
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, CANVAS_BG, FIXED_PARAMS, CENTER_X,
    SHAPE_OPTIONS, LAYOUT_OPTIONS, COLOR_OPTIONS, FONT_OPTIONS, INTERACTION_MODES,
    INITIAL_PARAMS, ADAPTABLE_PARAMS, HISTORY_LENGTH, DEFAULT_NUM_ITEMS,
    Q_TABLE_PATH, GPR_MODEL_PATH # Add GPR_MODEL_PATH for reset
)

from menu_manager import MenuManager
from rl_agent import PersonalizationAgent
from performance_monitor import PerformanceMonitor
from user_model import UserModelGPR # Import GPR model class

class VRPersonalizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Adaptive VR Personalizer Simulation")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.config = INITIAL_PARAMS.copy()
        self.canvas = None
        self.menu_manager = None
        self.agent = None
        self.perf_monitor = None
        self.user_model = None
        self.info_labels = {}
        self.current_target_item_index = None
        self.current_target_canvas_id = None
        self.trial_count = 0
        self.max_trials = 100
        self.is_running = False
        self.dialog_confirmed = False
        self.last_trajectory_features = {}
        self.last_gpr_prediction = {}
        self.fixed_num_items = DEFAULT_NUM_ITEMS

        # Plotting History Initialization
        self.plot_history = {
            'trial': [], 'scalar_reward': [], 'time_taken': [],
            'click_error': [], 'path_efficiency': [], 'jerk_metric': []
        }

        self.draw_voronoi_var = BooleanVar(value=False)
        self.setup_gui()
        if not self.show_setup_dialog():
             print("Setup cancelled. Using default settings.")
             self.config = INITIAL_PARAMS.copy()
             self.fixed_num_items = self.config.get('num_items', DEFAULT_NUM_ITEMS)
             if 'num_items' in self.config: del self.config['num_items']
             self.dialog_confirmed = True
        if self.dialog_confirmed:
             self.initialize_components()
             self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
             self.canvas.bind("<Motion>", self.handle_mouse_motion)
        else:
            self.root.quit()

    # --- METHOD DEFINITIONS ---

    def plot_results(self):
        # (Keep plotting method as is)
        if not self.plot_history['trial']: messagebox.showinfo("Plot Results", "No trial data available to plot."); return
        if not MATPLOTLIB_AVAILABLE: messagebox.showerror("Error", "Matplotlib library not found. Please install it (`pip install matplotlib`)."); return
        trials = self.plot_history['trial']; num_plots = 4
        fig, axs = plt.subplots(num_plots, 1, figsize=(7, 9), sharex=True)
        if num_plots == 1: axs = [axs]
        fig.suptitle('Performance Over Trials')
        axs[0].plot(trials, self.plot_history['scalar_reward'], marker='.', linestyle='-', label='Scalar Reward'); axs[0].set_ylabel('Reward'); axs[0].grid(True); axs[0].legend()
        color_time = 'tab:blue'; axs[1].plot(trials, self.plot_history['time_taken'], marker='.', linestyle='-', color=color_time, label='Time (s)'); axs[1].set_ylabel('Time (s)', color=color_time); axs[1].tick_params(axis='y', labelcolor=color_time); axs[1].grid(True)
        ax1_twin = axs[1].twinx(); color_error = 'tab:red'; ax1_twin.plot(trials, self.plot_history['click_error'], marker='.', linestyle='--', color=color_error, label='Error (px)'); ax1_twin.set_ylabel('Error (px)', color=color_error); ax1_twin.tick_params(axis='y', labelcolor=color_error)
        lines, labels = axs[1].get_legend_handles_labels(); lines2, labels2 = ax1_twin.get_legend_handles_labels(); axs[1].legend(lines + lines2, labels + labels2, loc='best')
        color_path = 'tab:green'; axs[2].plot(trials, self.plot_history['path_efficiency'], marker='.', linestyle='-', color=color_path, label='Path Efficiency'); axs[2].set_ylabel('Path Efficiency (0-1)'); axs[2].set_ylim(0, 1.1); axs[2].grid(True); axs[2].legend()
        color_jerk = 'tab:purple'; jerk_data = np.array(self.plot_history['jerk_metric']); valid_jerk = jerk_data[jerk_data > 0]; valid_trials = np.array(trials)[jerk_data > 0]
        if len(valid_jerk) > 0: axs[3].semilogy(valid_trials, valid_jerk, marker='.', linestyle='-', color=color_jerk, label='Jerk Metric (Log Scale)'); axs[3].set_ylabel('Jerk (Log Scale)')
        else: axs[3].plot(trials, jerk_data, marker='.', linestyle='-', color=color_jerk, label='Jerk Metric (Linear Scale)'); axs[3].set_ylabel('Jerk (Linear Scale)')
        axs[3].grid(True); axs[3].legend(); axs[-1].set_xlabel('Trial Number'); fig.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.show(block=False)

    # --- ADD EXPORT FUNCTION ---
    def export_results_to_excel(self):
        """Exports the collected trial data from plot_history to an Excel file."""
        if not self.plot_history or not self.plot_history['trial']:
            print("No history data to export.")
            messagebox.showwarning("Export Results", "No trial data recorded yet.")
            return

        try:
            # Create a DataFrame from the history dictionary
            df = pd.DataFrame(self.plot_history)

            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ask for participant ID (optional, for user studies)
            participant_id = simpledialog.askstring("Participant ID", "Enter Participant ID (optional, leave blank if not needed):", parent=self.root)
            if participant_id:
                filename = f"results_participant_{participant_id}_{timestamp}.xlsx"
            else:
                 filename = f"results_{timestamp}.xlsx"

            # Specify output directory (e.g., 'results' subfolder)
            output_dir = 'results'
            os.makedirs(output_dir, exist_ok=True)
            full_path = os.path.join(output_dir, filename)

            # Save DataFrame to Excel
            df.to_excel(full_path, index=False, engine='openpyxl') # index=False prevents writing the DataFrame index

            print(f"Results exported successfully to {full_path}")
            messagebox.showinfo("Export Successful", f"Results saved to:\n{full_path}")

        except ImportError:
             messagebox.showerror("Export Error", "Pandas or openpyxl library not found.\nPlease install them (`pip install pandas openpyxl`).")
        except Exception as e:
            print(f"Error exporting results to Excel: {e}")
            messagebox.showerror("Export Error", f"Could not save results to Excel:\n{e}")
    # --- END EXPORT FUNCTION ---


    def setup_gui(self):
        # (GUI setup remains the same, including the plot button)
        self.canvas = tk.Canvas(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT - 200, bg=CANVAS_BG); self.canvas.pack(pady=5); self.canvas.bind("<Button-1>", self.handle_canvas_click)
        info_frame = Frame(self.root); info_frame.pack(fill=tk.X, pady=5); info_frame.columnconfigure(0, weight=1); info_frame.columnconfigure(1, weight=1); info_frame.columnconfigure(2, weight=1); info_frame.columnconfigure(3, weight=1)
        self.info_labels['status'] = Label(info_frame, text="Status: Initializing", fg="blue", anchor='w'); self.info_labels['trial'] = Label(info_frame, text="Trial: 0/0", anchor='e')
        self.info_labels['status'].grid(row=0, column=0, columnspan=2, sticky='w', padx=10); self.info_labels['trial'].grid(row=0, column=3, sticky='e', padx=10)
        self.info_labels['params_core'] = Label(info_frame, text="Params: -", anchor='w'); self.info_labels['params_core'].grid(row=1, column=0, columnspan=4, sticky='w', padx=10)
        self.info_labels['params_layout'] = Label(info_frame, text="Layout Params: -", anchor='w'); self.info_labels['params_layout'].grid(row=2, column=0, columnspan=4, sticky='w', padx=10)
        self.info_labels['avg_perf'] = Label(info_frame, text="Avg Perf: -", anchor='w'); self.info_labels['avg_perf'].grid(row=3, column=0, columnspan=2, sticky='w', padx=10)
        self.info_labels['last_perf'] = Label(info_frame, text="Last Perf: -", anchor='w'); self.info_labels['last_perf'].grid(row=3, column=2, columnspan=2, sticky='w', padx=10)
        self.info_labels['gpr_info'] = Label(info_frame, text="GPR: -", anchor='w'); self.info_labels['gpr_info'].grid(row=4, column=0, columnspan=4, sticky='w', padx=10)
        control_frame = Frame(self.root); control_frame.pack(fill=tk.X, pady=10)
        self.start_button = tk.Button(control_frame, text="Start Trial", command=self.start_trial_sequence, width=12); self.start_button.pack(side=tk.LEFT, padx=10)
        self.reset_button = tk.Button(control_frame, text="Reset Settings", command=self.reset_settings, width=12); self.reset_button.pack(side=tk.LEFT, padx=10)
        self.save_button = tk.Button(control_frame, text="Save All", command=self.save_state_explicitly, width=10); self.save_button.pack(side=tk.LEFT, padx=10)
        self.load_button = tk.Button(control_frame, text="Load All", command=self.load_state_explicitly, width=10); self.load_button.pack(side=tk.LEFT, padx=10)
        self.plot_button = tk.Button(control_frame, text="Plot Results", command=self.plot_results, width=12); self.plot_button.pack(side=tk.LEFT, padx=10)
        # --- Add Export Button ---
        self.export_button = tk.Button(control_frame, text="Export Results", command=self.export_results_to_excel, width=12)
        self.export_button.pack(side=tk.LEFT, padx=10)
        # ---
        self.voronoi_check = Checkbutton(control_frame, text="Draw Voronoi", variable=self.draw_voronoi_var, command=self.redraw_menu); self.voronoi_check.pack(side=tk.LEFT, padx=10)


    def show_setup_dialog(self):
        # (Keep setup dialog as is)
        dialog = tk.Toplevel(self.root); dialog.title("Initial Setup"); dialog.geometry("380x380")
        dialog.transient(self.root); dialog.grab_set(); dialog.protocol("WM_DELETE_WINDOW", lambda: self.on_dialog_cancel(dialog))
        r = 1; tk.Label(dialog, text="Configure Menu Settings:").grid(row=0, column=0, columnspan=2, pady=10)
        tk.Label(dialog, text="Number of Items:").grid(row=r, column=0, sticky=tk.W, padx=5)
        self.num_items_var = tk.IntVar(value=self.config.get('num_items', self.fixed_num_items)) # Use current fixed value
        tk.Entry(dialog, textvariable=self.num_items_var, width=5).grid(row=r, column=1, sticky=tk.W); r+=1
        tk.Label(dialog, text="Initial Shape:").grid(row=r, column=0, sticky=tk.W, padx=5); self.shape_var = tk.StringVar(value=self.config.get('shape', SHAPE_OPTIONS[0])); tk.OptionMenu(dialog, self.shape_var, *SHAPE_OPTIONS).grid(row=r, column=1, sticky=tk.W); r+=1
        tk.Label(dialog, text="Initial Layout:").grid(row=r, column=0, sticky=tk.W, padx=5); self.layout_var = tk.StringVar(value=self.config.get('layout_algorithm', LAYOUT_OPTIONS[0])); tk.OptionMenu(dialog, self.layout_var, *LAYOUT_OPTIONS).grid(row=r, column=1, sticky=tk.W); r+=1
        tk.Label(dialog, text="Item Color:").grid(row=r, column=0, sticky=tk.W, padx=5); current_color_name = next((k for k, v in COLOR_OPTIONS.items() if v == self.config.get('color','lightblue')), list(COLOR_OPTIONS.keys())[0]); self.color_name_var = tk.StringVar(value=current_color_name); tk.OptionMenu(dialog, self.color_name_var, *COLOR_OPTIONS.keys()).grid(row=r, column=1, sticky=tk.W); r+=1
        tk.Label(dialog, text="Item Font:").grid(row=r, column=0, sticky=tk.W, padx=5); font_display_options = [f"{f[0]} {f[1]}" for f in FONT_OPTIONS]; cf = self.config.get('font', FONT_OPTIONS[0]); current_font_display = f"{cf[0]} {cf[1]}"; self.font_display_var = tk.StringVar(value=current_font_display); tk.OptionMenu(dialog, self.font_display_var, *font_display_options).grid(row=r, column=1, sticky=tk.W); r+=1
        tk.Label(dialog, text="Interaction Mode:").grid(row=r, column=0, sticky=tk.W, padx=5); self.mode_var = tk.StringVar(value=self.config.get('interaction_mode', INTERACTION_MODES[0])); tk.OptionMenu(dialog, self.mode_var, *INTERACTION_MODES).grid(row=r, column=1, sticky=tk.W); r+=1
        btn_frame = Frame(dialog); btn_frame.grid(row=r, column=0, columnspan=2, pady=20)
        ok_button = tk.Button(btn_frame, text="OK", command=lambda: self.on_dialog_ok(dialog), width=10); ok_button.pack(side=tk.LEFT, padx=10)
        cancel_button = tk.Button(btn_frame, text="Cancel", command=lambda: self.on_dialog_cancel(dialog), width=10); cancel_button.pack(side=tk.LEFT, padx=10)
        self.dialog_confirmed = False; self.root.wait_window(dialog); return self.dialog_confirmed

    def on_dialog_ok(self, dialog):
        # (Keep this method as it was, setting self.fixed_num_items)
        try:
            num_items = self.num_items_var.get()
            if not (4 <= num_items <= 16): raise ValueError("Number of items must be between 4 and 16.")
            self.config['shape'] = self.shape_var.get(); self.config['layout_algorithm'] = self.layout_var.get()
            self.config['color'] = COLOR_OPTIONS[self.color_name_var.get()]
            font_str = self.font_display_var.get().split(); self.config['font'] = (font_str[0], int(font_str[1]))
            self.config['interaction_mode'] = self.mode_var.get()
            self.fixed_num_items = num_items # Set the fixed value
            print(f"INFO: Number of items set to {self.fixed_num_items} in on_dialog_ok")
            for key in ADAPTABLE_PARAMS:
                 if key in self.config: INITIAL_PARAMS[key] = self.config[key]
            for key in FIXED_PARAMS: # Update fixed display defaults
                 if key in self.config: INITIAL_PARAMS[key] = self.config[key]
            # Ensure INITIAL_PARAMS doesn't accidentally contain num_items
            if 'num_items' in INITIAL_PARAMS: del INITIAL_PARAMS['num_items']
            self.dialog_confirmed = True; dialog.destroy()
        except ValueError as e: messagebox.showerror("Input Error", str(e), parent=dialog)
        except Exception as e: messagebox.showerror("Error", f"Failed to apply settings: {e}", parent=dialog)

    def on_dialog_cancel(self, dialog):
        self.dialog_confirmed = False
        dialog.destroy()

    def initialize_components(self):
        # (Keep this method as it was)
        print("Initializing components...")
        current_config_for_agent = {**ADAPTABLE_PARAMS, **self.config}
        if 'num_items' in current_config_for_agent: del current_config_for_agent['num_items']
        self.menu_manager = MenuManager(self.canvas); self.perf_monitor = PerformanceMonitor()
        N_GPR_FEATURES = 8; self.user_model = UserModelGPR(n_features=N_GPR_FEATURES)
        agent_init_keys = ADAPTABLE_PARAMS.keys() | {'interaction_mode'}
        agent_init_params = {k: current_config_for_agent[k] for k in agent_init_keys if k in current_config_for_agent}
        agent_init_params['num_items'] = self.fixed_num_items # Pass fixed num to agent init
        print(f"DEBUG (initialize_components): Initializing agent with params: {agent_init_params}")
        self.agent = PersonalizationAgent(agent_init_params, n_gpr_features=N_GPR_FEATURES)
        self.config.update(self.agent.current_params); # Update app config (adaptable params)
        if 'num_items' in self.config: del self.config['num_items'] # Ensure not in adaptable config
        self.menu_manager.update_parameters(self.agent.current_params, self.fixed_num_items)
        self.redraw_menu(); self.update_info_display(); self.update_status("Ready. Press 'Start Trial'.")
        print("Components initialized.")

    def reset_settings(self):
        # (Keep this method as it was, including plot history clearing)
        if self.is_running: messagebox.showwarning("Reset", "Cannot reset while a trial is running."); return
        old_config = self.config.copy(); old_fixed_num_items = self.fixed_num_items
        if self.show_setup_dialog():
            clear_learning = messagebox.askyesno("Clear Learning?", "Clear existing Q-Table and User Model data?")
            if clear_learning:
                 try:
                     if os.path.exists(Q_TABLE_PATH): os.remove(Q_TABLE_PATH)
                     if os.path.exists(GPR_MODEL_PATH): os.remove(GPR_MODEL_PATH)
                     print("Learning data cleared.")
                 except OSError as e: print(f"Error clearing learning data: {e}"); messagebox.showwarning("File Error", f"Could not delete learning files:\n{e}")
            self.trial_count = 0; self.last_trajectory_features = {}; self.last_gpr_prediction = {}
            for key in self.plot_history: self.plot_history[key] = []
            self.initialize_components()
            self.start_button.config(text="Start Trial", state=tk.NORMAL)
        else:
            self.config = old_config; self.fixed_num_items = old_fixed_num_items
            self.update_status("Reset cancelled.")

    def save_state_explicitly(self):
         # (Keep this method as it was)
         if self.agent: self.agent.save_q_table(); self.update_status("State saved.")
         else: self.update_status("Agent not initialized.")

    def load_state_explicitly(self):
         # (Keep this method as it was, num_items handling is in agent.load_q_table)
         if self.agent:
              fixed_num = self.fixed_num_items
              self.agent.load_q_table()
              self.agent.fixed_num_items = fixed_num; self.agent.current_params['num_items'] = fixed_num
              self.config.update(self.agent.current_params);
              if 'num_items' in self.config: del self.config['num_items']
              self.menu_manager.update_parameters(self.agent.current_params, self.fixed_num_items);
              self.redraw_menu()
              self.update_status("State loaded."); self.update_info_display()
         else: self.update_status("Agent not initialized.")

    def redraw_menu(self):
        """ Utility to redraw menu, respecting Voronoi toggle """
        if self.menu_manager and self.agent:
             params_for_draw = self.agent.current_params
             print(f"DEBUG (redraw_menu): Drawing with fixed_num_items = {self.fixed_num_items}")
             self.root.update_idletasks()
             self.menu_manager.update_parameters(params_for_draw, self.fixed_num_items) # Pass both
             self.menu_manager.draw_menu(draw_voronoi=self.draw_voronoi_var.get())

    def start_trial_sequence(self):
        # (Keep this method as it was)
        if self.is_running: return
        if self.trial_count >= self.max_trials: messagebox.showinfo("Finished", "Adaptation sequence already completed. Reset settings to run again."); return
        if not all([self.agent, self.menu_manager, self.perf_monitor, self.user_model]): messagebox.showerror("Error", "Components not initialized. Please reset settings."); return
        self.is_running = True; self.start_button.config(text="Running...", state=tk.DISABLED)
        self.update_status(f"Trial {self.trial_count + 1}/{self.max_trials}..."); self.run_next_trial()


    def run_next_trial(self):
        """Sets up and runs a single interaction trial."""
        if not self.is_running: return

        # --- Ensure menu manager uses fixed num items ---
        print(f"DEBUG (run_next_trial): Using fixed_num_items = {self.fixed_num_items}")
        self.menu_manager.update_parameters(self.agent.current_params, self.fixed_num_items)
        # ---

        self.redraw_menu()

        item_positions = self.menu_manager.get_item_positions()
        num_items_available = len(item_positions)
        if not item_positions or num_items_available != self.fixed_num_items:
            print(f"ERROR (run_next_trial): Incorrect number of items generated! Expected {self.fixed_num_items}, Got {num_items_available}")
            self.update_status("Error: Layout generated incorrect number of items."); self.is_running = False; return

        self.current_target_item_index = random.randrange(num_items_available)
        self.current_target_canvas_id = self.menu_manager.highlight_target(self.current_target_item_index)
        if self.current_target_canvas_id: self.perf_monitor.start()
        else: self.update_status(f"Error highlighting target index {self.current_target_item_index}."); self.is_running = False; self.start_button.config(state=tk.NORMAL)


    def handle_mouse_motion(self, event):
        # (Keep this method as it was)
        if self.is_running and self.perf_monitor: self.perf_monitor.record_motion(event)

    def handle_canvas_click(self, event):
        """Handles clicks on the canvas, determines hit item based on drawn position."""
        if self.current_target_item_index is None or not self.menu_manager or not self.agent:
            return

        clicked_item_index = -1
        detection_method = "None"

        # 1. Voronoi Check (if enabled) - Uses calculated points, consistent with cell generation
        if self.draw_voronoi_var.get():
            voronoi_cells = self.menu_manager.get_voronoi_cells()
            item_positions = self.menu_manager.get_item_positions() # Need original points for Voronoi mapping
            if voronoi_cells and len(voronoi_cells) == len(item_positions):
                detection_method = "Voronoi"
                for item_id, vertices in voronoi_cells.items():
                    if vertices and len(vertices) >= 3:
                        if point_in_polygon(event.x, event.y, vertices):
                            # Ensure the item_id from Voronoi is valid
                            if 0 <= item_id < len(item_positions):
                                clicked_item_index = item_id
                                break # Found hit via Voronoi

        # 2. Fallback: Minimum Distance Check (using DRAWN coordinates)
        if clicked_item_index == -1:
            detection_method = "Min Distance (Drawn Coords)"
            min_dist_sq = float('inf')
            best_hit_index = -1
            item_size = self.agent.current_params.get('item_size', 30)
            # Allow clicking slightly outside the visual edge
            max_allowed_dist_sq = (item_size / 2 + 10)**2 # Tolerance

            # Get all drawn item canvas IDs
            drawn_item_canvas_ids = self.canvas.find_withtag(self.menu_manager.MENU_ITEM_TAG)

            for canvas_id in drawn_item_canvas_ids:
                # Extract the original item index (0-based) from the tag
                tags = self.canvas.gettags(canvas_id)
                item_index = -1
                for tag in tags:
                    if tag.startswith("item_") and not tag.startswith("item_text_"):
                        try:
                            item_index = int(tag.split("_")[1])
                            break
                        except (IndexError, ValueError):
                            continue # Ignore malformed tags or text tags

                if item_index == -1:
                    # print(f"Warning: Could not extract index from tags: {tags} for canvas_id: {canvas_id}")
                    continue # Skip if we can't identify the item index

                # Get the ACTUAL drawn coordinates (bounding box)
                try:
                    coords = self.canvas.coords(canvas_id)
                    if not coords or len(coords) < 4:
                        # print(f"Warning: Could not get valid coords for canvas_id: {canvas_id}")
                        continue # Skip if coords are invalid
                except tk.TclError:
                    # print(f"Warning: TclError getting coords for canvas_id: {canvas_id} (item might be deleted)")
                    continue # Skip if item doesn't exist

                # Calculate the center of the DRAWN item
                drawn_x, drawn_y = get_midpoint(coords)

                # Calculate distance from click to the drawn center
                dist_sq = (event.x - drawn_x)**2 + (event.y - drawn_y)**2

                # Check if within tolerance and closer than previous best
                if dist_sq < min_dist_sq and dist_sq <= max_allowed_dist_sq:
                    min_dist_sq = dist_sq
                    best_hit_index = item_index # Store the 0-based index

            # If a hit was found within tolerance, assign it
            if best_hit_index != -1:
                clicked_item_index = best_hit_index

        # --- End of Click Detection Logic ---

        print(f"FINAL DEBUG Click: Method={detection_method}, Detected Index={clicked_item_index}, Target Index={self.current_target_item_index}")

        if not self.is_running:
            return # Don't process clicks if trial isn't active

        # --- Process the Click Result ---
        if clicked_item_index == self.current_target_item_index:
            # --- CORRECT TARGET HIT ---
            target_canvas_id_to_use = self.current_target_canvas_id
            # Double-check if the stored ID is valid, find it again if needed
            if not target_canvas_id_to_use or not self.canvas.coords(target_canvas_id_to_use):
                print("Warning: Stored target canvas ID was invalid or missing. Refetching...")
                ids = self.canvas.find_withtag(f"item_{self.current_target_item_index}")
                if not ids:
                    print(f"CRITICAL ERROR: Cannot find target item {self.current_target_item_index} on canvas after hit detection!")
                    self.update_status("Internal Error: Target item missing.")
                    self.is_running = False; self.start_button.config(state=tk.NORMAL)
                    return # Cannot proceed
                target_canvas_id_to_use = ids[0]
                self.current_target_canvas_id = target_canvas_id_to_use # Update stored ID

            performance = self.perf_monitor.stop(event, target_canvas_id_to_use, self.canvas)
            if performance:
                # --- RL & GPR Update ---
                self.last_trajectory_features = {'path_efficiency': performance.get('path_efficiency', 0), 'jerk_metric': performance.get('jerk_metric', 1e6)}
                avg_perf = self.perf_monitor.get_average_performance()
                params_for_model = {k:v for k,v in self.agent.current_params.items() if k != 'num_items'}
                self.user_model.update(params_for_model, performance, avg_perf, self.last_trajectory_features)
                self.last_gpr_prediction = self.user_model.predict(params_for_model, avg_perf, self.last_trajectory_features)
                current_state = self.agent.get_state(avg_perf, self.last_trajectory_features, self.last_gpr_prediction)
                reward_vector = self.perf_monitor.calculate_reward_vector(performance, self.config['interaction_mode'])
                scalar_reward = self.perf_monitor.scalarize_reward(reward_vector)
                action_index = self.agent.choose_action(current_state, self.last_gpr_prediction)
                new_params = self.agent.apply_action_and_get_new_params(action_index) # Agent ensures num_items fixed
                params_for_next_pred = {k:v for k,v in new_params.items() if k != 'num_items'}
                next_gpr_pred = self.user_model.predict(params_for_next_pred, avg_perf, self.last_trajectory_features)
                next_state = self.agent.get_state(avg_perf, self.last_trajectory_features, next_gpr_pred)
                self.agent.update(current_state, action_index, scalar_reward, next_state)
                # Record data for plotting
                current_trial_num = self.trial_count + 1
                self.plot_history.setdefault('trial', []).append(current_trial_num)
                self.plot_history.setdefault('scalar_reward', []).append(scalar_reward)
                self.plot_history.setdefault('time_taken', []).append(performance['time_taken'])
                self.plot_history.setdefault('click_error', []).append(performance['click_error'])
                self.plot_history.setdefault('path_efficiency', []).append(performance.get('path_efficiency', 0))
                self.plot_history.setdefault('jerk_metric', []).append(performance.get('jerk_metric', 0))
                # Update GUI
                self.redraw_menu(); self.update_info_display()
                # Prepare for next trial or end
                self.trial_count += 1; self.current_target_item_index = None; self.current_target_canvas_id = None
                if self.trial_count < self.max_trials:
                    self.update_status(f"Trial {self.trial_count + 1}/{self.max_trials}. Waiting...")
                    self.root.after(150, self.run_next_trial)
                else:
                    self.is_running = False; self.start_button.config(text="Finished", state=tk.DISABLED)
                    self.update_status(f"Adaptation complete ({self.max_trials} trials). Exporting results...")
                    final_params = {k:v for k,v in self.agent.current_params.items() if k != 'num_items'}
                    # --- Automatically export results ---
                    self.export_results_to_excel()
                    # ---
                    messagebox.showinfo("Finished", f"Adaptation finished ({self.max_trials} trials).\n"
                                            f"Results exported to Excel.\n"
                                            f"Window remains open. Close manually to save final RL/GPR state.\n\n"
                                            f"Final Params: Size={final_params['item_size']:.1f}, "
                                            f"Radius={final_params['layout_radius']:.1f}, "
                                            f"Shape={final_params.get('shape','N/A')}, "
                                            f"Layout={final_params.get('layout_algorithm','N/A')}")
            else: # Performance measurement failed
                self.update_status("Error measuring performance.")
                self.is_running = False; self.start_button.config(state=tk.NORMAL)
        elif clicked_item_index != -1: # Clicked, but wrong item
            self.update_status(f"Wrong item clicked (Detected: {clicked_item_index+1}). Target is {self.current_target_item_index+1}. Waiting...")
        else: # Clicked empty space
             self.update_status(f"Missed. Target is {self.current_target_item_index+1}. Waiting...")


    def update_info_display(self):
        # (Keep this method as it was)
        if not self.agent or not self.perf_monitor or not self.user_model: return
        cp = self.agent.current_params; core_str = (f"Size: {cp['item_size']:.1f} | Radius: {cp['layout_radius']:.1f} | Shape: {cp.get('shape','?')} | Layout: {cp.get('layout_algorithm','?')}")
        self.info_labels['params_core'].config(text=core_str); layout = cp.get('layout_algorithm'); layout_params_str = f"Layout Params ({layout}): "
        if layout == 'force_directed': layout_params_str += f"K_rep={cp.get('force_k_repel',0):.0f}, K_att={cp.get('force_k_attract',0):.2f}"
        elif layout == 'ergonomic_arc': layout_params_str += f"Offset=({cp.get('arc_center_offset_x',0)}, {cp.get('arc_center_offset_y',0)}), Angles=[{cp.get('arc_start_angle',0):.2f}, {cp.get('arc_end_angle',0):.2f}]"
        else: layout_params_str += "N/A"
        self.info_labels['params_layout'].config(text=layout_params_str); avg_p = self.perf_monitor.get_average_performance()
        avg_perf_str = (f"Avg Perf (last {min(self.trial_count, HISTORY_LENGTH)}): T={avg_p['avg_time']:.2f}s, E={avg_p['avg_error']:.1f}px, PathEff={avg_p.get('avg_path_efficiency',0):.2f}, Jerk={avg_p.get('avg_jerk',0):.1e}")
        self.info_labels['avg_perf'].config(text=avg_perf_str); last_p = self.perf_monitor.last_performance
        if last_p: last_perf_str = (f"Last Perf: T={last_p['time_taken']:.2f}s (Pred:{last_p.get('predicted_time',0):.2f}), E={last_p['click_error']:.1f}px, PathEff={last_p.get('path_efficiency',0):.2f}, Jerk={last_p.get('jerk_metric',0):.1e}")
        else: last_perf_str = "Last Perf: -"
        self.info_labels['last_perf'].config(text=last_perf_str); gpr_pred = self.last_gpr_prediction
        if gpr_pred: gpr_str = (f"GPR Pred: T={gpr_pred.get('pred_time',0):.2f}s, E={gpr_pred.get('pred_error',0):.1f}px | Uncertainty={gpr_pred.get('uncertainty',0):.3f} | Fitted: {self.user_model.is_fitted}")
        else: gpr_str = "GPR: Not run yet."
        self.info_labels['gpr_info'].config(text=gpr_str); self.info_labels['trial'].config(text=f"Trial: {self.trial_count}/{self.max_trials}")

    def update_status(self, message):
        # (Keep this method as it was)
        self.info_labels['status'].config(text=f"Status: {message}")

    def on_closing(self):
        # (Keep this method as it was)
        print("Closing application and saving state...")
        if self.agent: self.agent.save_q_table()
        self.root.destroy()


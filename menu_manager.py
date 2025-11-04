# menu_manager.py
import tkinter as tk
import math
import numpy as np
# Import necessary functions and constants
from utils import (calculate_circular_or_ellipse_positions, calculate_clustered_positions,
                   calculate_arc_positions, get_regular_polygon_vertices, get_voronoi_cells, point_in_polygon)
from layout_engine import calculate_force_directed_layout
# Import dimensions for clamping
from config import CENTER_X, CENTER_Y, CANVAS_BG, WINDOW_WIDTH, DRAWING_AREA_HEIGHT

class MenuManager:
    MENU_ITEM_TAG = "menu_item"
    TARGET_TAG = "target_item"
    VORONOI_CELL_TAG = "voronoi_cell"

    def __init__(self, canvas):
        self.canvas = canvas
        self.current_params = {}
        self.item_canvas_ids = []
        self.item_positions = []
        self.voronoi_cells = {}
        self.voronoi_obj = None

    def update_parameters(self, params, fixed_num_items):
        # (Keep update_parameters method as in the previous correct version)
        self.current_params = params.copy()
        shape_name = self.current_params.get('shape', 'circle')
        sides_map = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6}
        self.current_params['polygon_sides'] = sides_map.get(shape_name, 0)
        layout_algo = self.current_params.get('layout_algorithm', 'circular')
        radius = self.current_params.get('layout_radius', 150)
        print(f"DEBUG (update_parameters): Algorithm='{layout_algo}', Fixed NumItems={fixed_num_items}, Radius={radius:.1f}")
        if layout_algo == 'circular': self.item_positions = calculate_circular_or_ellipse_positions(CENTER_X, CENTER_Y, fixed_num_items, radius, self.current_params)
        elif layout_algo == 'clustered_dominant_hand': self.item_positions = calculate_clustered_positions(CENTER_X, CENTER_Y, fixed_num_items, radius, self.current_params)
        elif layout_algo == 'ergonomic_arc': self.item_positions = calculate_arc_positions(CENTER_X, CENTER_Y, fixed_num_items, radius, self.current_params)
        elif layout_algo == 'force_directed': current_pos_data = self.item_positions if len(self.item_positions) == fixed_num_items else None; self.item_positions = calculate_force_directed_layout(fixed_num_items, current_pos_data, self.current_params)
        else: print(f"Warning: Unknown layout algorithm '{layout_algo}'. Defaulting to circular."); self.item_positions = calculate_circular_or_ellipse_positions(CENTER_X, CENTER_Y, fixed_num_items, radius, self.current_params)
        calculated_pos_count = len(self.item_positions)
        print(f"DEBUG (update_parameters): Layout function returned {calculated_pos_count} positions.")
        if calculated_pos_count != fixed_num_items: print(f"ERROR: Layout function returned wrong number of positions! Expected {fixed_num_items}, got {calculated_pos_count}.")
        points_for_voronoi = [(p['x'], p['y']) for p in self.item_positions]
        if len(points_for_voronoi) > 0:
             canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
             if canvas_width <= 1 or canvas_height <= 1: print("Warning: Canvas dimensions not ready for Voronoi calculation."); self.voronoi_cells, self.voronoi_obj = {}, None
             else: self.voronoi_cells, self.voronoi_obj = get_voronoi_cells(points_for_voronoi, canvas_width, canvas_height)
        else: self.voronoi_cells, self.voronoi_obj = {}, None


    def draw_menu(self, draw_voronoi=False):
        self.canvas.delete(self.MENU_ITEM_TAG); self.canvas.delete(self.TARGET_TAG); self.canvas.delete(self.VORONOI_CELL_TAG)
        self.item_canvas_ids = []

        num_items_to_draw = len(self.item_positions)
        if not self.current_params or num_items_to_draw == 0:
            print("Warning: Cannot draw menu - params missing or zero positions calculated."); return

        shape = self.current_params.get('shape', 'circle')
        size = self.current_params.get('item_size', 30)
        color = self.current_params.get('color', 'lightblue')
        font_info = self.current_params.get('font', ("Arial", 10))
        sides = self.current_params.get('polygon_sides', 0)
        stretch = self.current_params.get('ellipse_stretch_ratio', 1.5)

        print(f"DEBUG (draw_menu): Attempting to draw {num_items_to_draw} items. Shape='{shape}', Sides={sides}")

        # Draw Voronoi cells if requested
        if draw_voronoi and self.voronoi_cells:
             # (Voronoi drawing logic...)
             for item_id, vertices in self.voronoi_cells.items():
                 if vertices and len(vertices) >= 3:
                      flat_vertices = [coord for point in vertices for coord in point]
                      try: self.canvas.create_polygon(flat_vertices, fill=CANVAS_BG, outline="lightgrey", dash=(2, 4), tags=self.VORONOI_CELL_TAG)
                      except tk.TclError as e: print(f"Warning: TclError drawing Voronoi cell {item_id}: {e}")

        # Draw Menu Items
        items_drawn_count = 0
        # --- Define Canvas Boundaries ---
        # Use a margin slightly larger than max item radius
        margin = (size / 2) + 5
        min_x_bound, max_x_bound = margin, WINDOW_WIDTH - margin
        min_y_bound, max_y_bound = margin, DRAWING_AREA_HEIGHT - margin
        # ---

        for item_info in self.item_positions:
            x, y = item_info['x'], item_info['y']
            item_id = item_info['id'] # 0-based index
            rad = size / 2
            canvas_id = None # Reset for each item
            item_tags = (self.MENU_ITEM_TAG, f"item_{item_id}")
            item_label = f"{item_id+1}" # 1-based label for display

            # Validate coordinates and size before drawing
            valid_coords = isinstance(x, (int, float)) and np.isfinite(x) and np.isfinite(y)
            valid_size = isinstance(rad, (int, float)) and np.isfinite(rad) and rad > 0.1
            valid_stretch = isinstance(stretch, (int, float)) and np.isfinite(stretch) and stretch > 0
            if not valid_coords or not valid_size: print(f"ERROR: Invalid coords/size for item_id={item_id}! Pos=({x}, {y}), Rad={rad}. Skipping."); continue
            if shape == 'ellipse' and not valid_stretch: print(f"ERROR: Invalid stretch for ellipse item_id={item_id}! Stretch={stretch}. Skipping."); continue

            # --- CLAMP COORDINATES TO CANVAS BOUNDS ---
            original_x, original_y = x, y
            x = max(min_x_bound, min(x, max_x_bound))
            y = max(min_y_bound, min(y, max_y_bound))
            if x != original_x or y != original_y:
                 print(f"DEBUG: Clamped item {item_id} coords from ({original_x:.1f},{original_y:.1f}) to ({x:.1f},{y:.1f})")
            # --- END CLAMPING ---

            # print(f"DEBUG Draw Loop: Item Index={item_id}, Label={item_label}, Pos=({x:.1f}, {y:.1f}), Rad={rad:.1f}, Tag='item_{item_id}'")

            # Draw the item shape
            try:
                if shape == 'circle': canvas_id = self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black", tags=item_tags)
                elif shape == 'ellipse':
                    hr, vr = rad * stretch, rad / stretch
                    if not (np.isfinite(hr) and np.isfinite(vr) and hr > 0 and vr > 0): print(f"ERROR: Invalid ellipse radii for item_id={item_id}! hr={hr}, vr={vr}. Skipping."); canvas_id = None
                    else: canvas_id = self.canvas.create_oval(x - hr, y - vr, x + hr, y + vr, fill=color, outline="black", tags=item_tags)
                elif shape == 'square': canvas_id = self.canvas.create_rectangle(x - rad, y - rad, x + rad, y + rad, fill=color, outline="black", tags=item_tags)
                elif sides >= 3:
                    vertices = get_regular_polygon_vertices(x, y, rad, sides) # Use clamped x, y
                    if vertices and len(vertices) >= 6 and all(isinstance(v, (int, float)) and np.isfinite(v) for v in vertices):
                        canvas_id = self.canvas.create_polygon(vertices, fill=color, outline="black", tags=item_tags)
                        if not canvas_id: print(f"ERROR: create_polygon FAILED for item_id={item_id}")
                    else: print(f"ERROR: Invalid vertices for polygon item_id={item_id}. Vertices: {vertices}"); canvas_id = None
                else: print(f"Warning: Unknown shape '{shape}' for item {item_id}.")

                if canvas_id:
                    font_size = font_info[1]
                    if size > font_size * 1.5: self.canvas.create_text(x, y, text=item_label, font=font_info, tags=(self.MENU_ITEM_TAG, f"item_text_{item_id}"), state=tk.DISABLED)
                    self.item_canvas_ids.append(canvas_id); items_drawn_count += 1
                else:
                    if shape in ['circle', 'ellipse', 'square'] or sides >= 3: print(f"ERROR: Failed to create canvas item for item_id={item_id} (Shape: {shape})")

            except tk.TclError as e: print(f"ERROR: TclError drawing item {item_id} ({shape}): {e}")
            except Exception as e: print(f"ERROR: Unexpected error drawing item {item_id} ({shape}): {e}")

        print(f"DEBUG (draw_menu): Finished drawing loop. Successfully drew {items_drawn_count} canvas items.")
        if items_drawn_count != num_items_to_draw:
             print(f"ERROR: Mismatch after drawing loop! Drew {items_drawn_count} items but expected {num_items_to_draw} based on positions.")
             print(f"DEBUG: Calculated positions: {self.item_positions}")
             print(f"DEBUG: Created canvas IDs: {self.item_canvas_ids}")


    def highlight_target(self, target_item_index):
        # (Highlight logic remains the same)
        self.canvas.delete(self.TARGET_TAG)
        if not (0 <= target_item_index < len(self.item_positions)): print(f"Warning: Invalid target index {target_item_index} requested."); return None
        target_tag = f"item_{target_item_index}"; canvas_ids = self.canvas.find_withtag(target_tag)
        if canvas_ids:
            target_canvas_id = canvas_ids[0]; text_tag = f"item_text_{target_item_index}"; text_ids = self.canvas.find_withtag(text_tag)
            self.canvas.itemconfig(target_canvas_id, fill="red", outline="yellow", width=2); self.canvas.addtag_withtag(self.TARGET_TAG, target_canvas_id)
            self.canvas.tag_raise(target_canvas_id);
            if text_ids: self.canvas.tag_raise(text_ids[0])
            print(f"DEBUG: Highlighted target index {target_item_index} (Label {target_item_index+1}, Canvas ID: {target_canvas_id})")
            return target_canvas_id
        else:
            all_tags = set();
            for item_id_on_canvas in self.canvas.find_withtag(self.MENU_ITEM_TAG): all_tags.update(self.canvas.gettags(item_id_on_canvas))
            if target_tag not in all_tags: print(f"ERROR: Target tag '{target_tag}' does not exist on canvas! Item likely failed to draw.")
            else: print(f"Warning: Could not find canvas item with tag '{target_tag}' to highlight.")
            return None

    def get_item_ids(self): return self.item_canvas_ids
    def get_item_positions(self): return self.item_positions
    def get_voronoi_cells(self): return self.voronoi_cells

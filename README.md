# automatic_VR_menue_designer

This paper presents an algorithmic complex for geometric modeling and dynamic adaptation of user interfaces in virtual reality (VR) environments, based on methods of engineering geometry and computer graphics.

This repository contains a **desktop simulation and research prototype** of that system: an automatic VR menu designer that learns how to arrange and parameterize menu elements so that users can work faster, more accurately and with less fatigue.

The implementation corresponds to the article:

> **“AN ALGORITHMIC COMPLEX FOR DYNAMIC ADAPTATION OF EDUCATIONAL ENVIRONMENT INTERFACES IN VIRTUAL REALITY BASED ON GEOMETRIC MODELING.”**

---

## Overview

The application models a VR pointing task on a 2D screen (mouse or touchpad).  
For each trial:

1. A menu of items is generated using geometric layout algorithms.
2. The user must select the highlighted target item.
3. The system records detailed performance and trajectory data.
4. A **Q-learning agent** updates interface parameters (size, shape, layout, etc.).
5. A **Gaussian Process Regression (GPR)** user model predicts future performance and provides uncertainty estimates to balance exploration/exploitation.

Over many trials, the system gradually converges to an ergonomic interface configuration tuned to the current user.

---

## Key features

- **Geometric menu layouts**
  - Circular / elliptical layouts
  - Ergonomic arc layout around the dominant hand
  - Clustered layout near the dominant-hand region
  - Force-directed layout with repulsive/attractive forces
  - Support for regular polygons (triangle, square, pentagon, hexagon)
  - Optional Voronoi visualization around menu items

- **Adaptive interface parameters**
  - Item size
  - Layout radius and position
  - Shape of menu elements
  - Layout algorithm selection
  - Force-directed layout parameters (`k_repel`, `k_attract`, optimal distance)
  - Arc center offsets and angle range
  - GPR exploration factor

- **Multi-objective performance metrics**
  - Task completion time
  - Click error / distance to target
  - Fitts’ law index of difficulty and predicted time
  - Path length and path efficiency (directness)
  - Jerk metric for movement smoothness
  - Optional distance penalty for simulated one-handed interaction

- **Reinforcement learning + user modeling**
  - Tabular Q-learning with ε-greedy policy
  - Discretized state: current parameters + averaged performance + trajectory features + GPR uncertainty
  - Vector reward → scalar reward using configurable weights
  - GPR models (time and error) with uncertainty-driven exploration

- **Visualization & data export**
  - Tkinter GUI with:
    - live summary of current parameters
    - average vs. last-trial performance
    - GPR predictions and uncertainty
    - trial counter and status messages
  - Optional matplotlib plots of:
    - reward over trials  
    - time & error  
    - path efficiency  
    - jerk metric
  - Export of all trial data to an Excel file (`results/*.xlsx`) for further analysis.

> In experimental validation with 17 students, the system significantly reduced task completion time and achieved high user satisfaction (94.2%), while improving key metrics such as jerk and path efficiency over the course of training.

---

## Repository structure

- `main.py` – Simple entry point that starts the Tkinter application.
- `app.py` – High-level **VRPersonalizerApp**:
  - builds the GUI,
  - orchestrates trials,
  - connects the RL agent, user model, menu manager and performance monitor,
  - manages plotting and export.
- `config.py` – All global configuration:
  - window sizes and canvas geometry,
  - reward weights and Fitts’ law constants,
  - RL hyperparameters and action set,
  - parameter bounds and default values,
  - available shapes, layouts, colors, fonts and interaction modes.
- `menu_manager.py` – Responsible for drawing the menu:
  - computes item positions via layout functions,
  - draws shapes on the Tkinter canvas,
  - maintains mappings between logical items and canvas IDs,
  - can optionally draw Voronoi cells.
- `layout_engine.py` – Force-directed layout algorithm and helpers.
- `performance_monitor.py` – Records user trajectories and computes:
  - time, error, Fitts metrics,
  - path efficiency,
  - jerk metric,
  - vector and scalar rewards.
- `user_model.py` – Gaussian Process Regression user model:
  - builds and updates GPR models for time and error,
  - uses a feature vector combining parameters, performance history and trajectory features,
  - can save/load the model to/from disk.
- `rl_agent.py` – Q-learning **PersonalizationAgent**:
  - maintains the Q-table and exploration rate,
  - discretizes the continuous state,
  - chooses actions, applies parameter changes with bounds,
  - saves/loads the Q-table and delegates GPR persistence.
- `utils.py` – Geometry and analysis utilities:
  - distances, midpoints,
  - regular polygon vertices,
  - Voronoi diagram construction and point-in-polygon,
  - trajectory feature extraction (path length, directness, jerk).


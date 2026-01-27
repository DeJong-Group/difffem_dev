import numpy as np
import json
import pyvista as pv


grid = pv.read('results/prism_geometry.vtu')

exp_names = [
    # these are node-based
    # "3d_rebar_cable",
    # "3d_rebar_full",
    # "3d_rebar_face",
    "3d_prism_rebar_fixed",
    "3d_rebar_cable_cell_cable",
    "3d_rebar_cable_cell_face",
    "3d_rebar_cable_cell_full",
    "3d_rebar_cable_cell_cable_quad",
    "3d_prism_slice_rebar",
    "3d_prism_slice_full",
    "3d_prism_slice_face",    
]

for exp_name in exp_names:
    print(f"Generating {exp_name}")
    # 1. Load the data
    with open(f"results/{exp_name}.json", "r") as f:
        data = json.load(f)

    E_history = np.array(data["E_hist"])  # List of arrays (one per iteration)
    num_iterations = len(E_history)


    # 3. Setup the Plotter for the GIF
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(f"figures/{exp_name}.gif")

    # Set consistent camera view and color limits
    vmin, vmax = 23.0e9, 27.0e9 

    # Add a title that we will update
    title = plotter.add_text(f"Iteration: 0", position='upper_left')

    # Initialize mesh
    plotter.add_mesh(
        grid, 
        scalars=E_history[0].ravel(),
        lighting=False,
        # cmap="jet", 
        clim=[vmin, vmax], 
        show_edges=True,
        scalar_bar_args={"title": "Young's Modulus (Pa)"}
    )

    print(f"Generating GIF with {num_iterations} frames...")

    # 4. Animation Loop
    # We skip frames if the history is too long (e.g., plot every 2nd step)
    step_size = 1 
    for i in range(0, num_iterations, step_size):
        # Update data
        grid.cell_data["values"] = E_history[i].ravel()
        plotter.update_scalars(E_history[i].ravel(), render=False)
        
        # Update text
        title.set_text(position="upper_left", text=f"Iteration: {i}")
        
        # Render and write frame
        plotter.write_frame()

    # Close and finalize
    plotter.close()
    print(f"Finished {exp_name}")
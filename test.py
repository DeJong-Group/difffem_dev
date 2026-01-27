import numpy as np
import pyvista

import warp as wp
import warp.fem as fem


@fem.integrand
def ackley(s: fem.Sample, domain: fem.Domain):
   x = domain(s)
   return (
      -20.0 * wp.exp(-0.2 * wp.sqrt(0.5 * wp.length_sq(x)))
      - wp.exp(0.5 * (wp.cos(2.0 * wp.pi * x[0]) + wp.cos(2.0 * wp.pi * x[1])))
      + wp.e
      + 20.0
   )


# Define field
geo = fem.Grid2D(res=wp.vec2i(64, 64), bounds_lo=wp.vec2(-4.0, -4.0), bounds_hi=wp.vec2(4.0, 4.0))
space = fem.make_polynomial_space(geo, degree=3)
field = space.make_field()
fem.interpolate(ackley, dest=field)

# Extract cells, nodes and values
cells, types = field.space.vtk_cells()
nodes = field.space.node_positions().numpy()
values = field.dof_values.numpy()
positions = np.hstack((nodes, values[:, np.newaxis]))
print(nodes.shape)
print(values.shape)
print(positions.shape)


# Visualize with pyvista
grid = pyvista.UnstructuredGrid(cells, types, positions)
grid.point_data["scalars"] = values
plotter = pyvista.Plotter()
plotter.add_mesh(grid)
plotter.show()
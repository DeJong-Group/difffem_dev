
import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.optim import Adam

@wp.func
def lame_from_E_nu(E: float, nu: float) -> wp.vec2:
    # λ = Eν / ((1+ν)(1−2ν)),  μ = E / (2(1+ν))
    lam = E * nu / ((1.0 + nu) * (1.0 - nu))
    mu = E / (2.0 * (1.0 + nu))
    return wp.vec2(lam, mu)

@fem.integrand(kernel_options={"max_unroll": 1})
def boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))

@fem.integrand
def classify_boundary_sides(
    s: fem.Sample,
    domain: fem.Domain,
    left: wp.array(dtype=int),
    right: wp.array(dtype=int),
):
    nor = fem.normal(domain, s)

    if nor[0] < -0.5:
        left[s.qp_index] = 1
    elif nor[0] > 0.5:
        right[s.qp_index] = 1

@wp.func
def hooke_stress(
    strain: wp.mat33, 
    lamb: wp.float32, 
    mu: wp.float32
):
    """Hookean elasticity"""
    return 2.0 * mu * strain + lamb * wp.trace(strain) * wp.identity(n=3, dtype=float)

@fem.integrand
def strain_field(s: fem.Sample, u: fem.Field):
    return fem.D(u, s)

@fem.integrand
def hooke_elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, E_field: fem.Field, nu: float):
    E_val = E_field(s)
    l = lame_from_E_nu(E_val, nu)
    lamb = l[0]
    mu = l[1]
    stress = hooke_stress(fem.D(u, s), lamb, mu)
    return wp.ddot(fem.D(v, s), stress)

@fem.integrand
def applied_load_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, load: wp.array(dtype=wp.float32)):
    return v(s)[0]*load[0] + v(s)[1]*load[1] + v(s)[2]*load[2]

@fem.integrand
def loss_disp(
    s: fem.Sample, domain: fem.Domain, u: fem.Field, u_meas: fem.Field
):
    disp = u(s)
    disp_meas = u_meas(s)
    diff = disp - disp_meas
    stress_norm_sq = 0.5 * wp.dot(diff, diff)

    return stress_norm_sq 

@fem.integrand
def loss_form(
    s: fem.Sample, domain: fem.Domain, u: fem.Field, u_meas: fem.Field
):
    strain = strain_field(s, u)
    strain_meas = strain_field(s, u_meas)
    diff = strain - strain_meas
    stress_norm_sq = 0.5 * wp.ddot(diff, diff) * 1e20

    return stress_norm_sq 



class Example:
    def __init__(
        self,
        quiet=False,
        degree=1,
        resolution=(200, 12, 12),
        mesh="tri",
        poisson_ratio=0.3,
        E=25.0e9,
        load=(1.0, 0.0, 0.0),
        lr=1.0e-3,
        strain_meas = None,
        u_meas = None,
    ):
        self._quiet = quiet
        self.degree = degree
        self.lr = lr
        # procedural rectangular domain definition
        bounds_lo = wp.vec3(0.0, 0.0, 0.0)
        bounds_hi = wp.vec3(2.0, 0.12, 0.12)
        self._initial_volume = (bounds_hi - bounds_lo)[0] * (bounds_hi - bounds_lo)[1] * (bounds_hi - bounds_lo)[2]
        
        self.strain_meas = strain_meas
        self.u_meas = u_meas

        if mesh == "tri":
            # triangle mesh, optimize vertices directly
            positions, tri_vidx = fem_example_utils.gen_tetmesh(
                res=wp.vec3i(resolution[0], resolution[1], resolution[2]), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            self._geo = fem.Tetmesh(tet_vertex_indices=tri_vidx, positions=positions)
            # self._start_geo = fem.Tetmesh(tri_vertex_indices=tri_vidx, positions=wp.clone(positions))
            self._vertex_positions = positions
        elif mesh == "quad":
            # quad mesh, optimize vertices directly
            positions, quad_vidx = fem_example_utils.gen_hexmesh(
                res=wp.vec3i(resolution[0], resolution[1], resolution[2]), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            self._geo = fem.Hexmesh(hex_vertex_indices=quad_vidx, positions=positions)
            # self._start_geo = fem.Hexmesh(quad_vertex_indices=quad_vidx, positions=wp.clone(positions))
            self._vertex_positions = positions
        else:
            # grid, optimize nodes of deformation field
            self._start_geo = fem.Grid3D(
                wp.vec3i(resolution[0], resolution[1], resolution[2]), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            vertex_displacement_space = fem.make_polynomial_space(self._start_geo, degree=degree, dtype=wp.vec3)
            vertex_position_field = fem.make_discrete_field(space=vertex_displacement_space)
            vertex_position_field.dof_values = vertex_displacement_space.node_positions()
            self._geo = vertex_position_field.make_deformed_geometry(relative=False)

        # make sure positions are differentiable
        # self._vertex_positions.requires_grad = True

        # Store initial node positions (for rendering)
        self._u_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=wp.vec3)
        self._start_node_positions = self._u_space.node_positions()

        # displacement field, make sure gradient is stored
        self._u_field = fem.make_discrete_field(space=self._u_space)
        self._u_field.dof_values.requires_grad = True

        self._u_field_meas = fem.make_discrete_field(space=self._u_space)
        self._u_field_meas.dof_values.requires_grad = True

        # Trial and test functions
        self._u_test = fem.make_test(space=self._u_space)
        self._u_trial = fem.make_trial(space=self._u_space)

        # Identify left and right sides for boundary conditions
        boundary = fem.Sides(self._geo)

        left_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        right_mask = wp.zeros(shape=boundary.element_count(), dtype=int)

        fem.interpolate(
            classify_boundary_sides,
            quadrature=fem.RegularQuadrature(boundary, order=0),
            values={"left": left_mask, "right": right_mask},
        )

        self._left = fem.Subdomain(boundary, element_mask=left_mask)
        self._right = fem.Subdomain(boundary, element_mask=right_mask)

        # Build projectors for the left-side homogeneous Dirichlet condition
        u_left_bd_test = fem.make_test(space=self._u_space, domain=self._left)
        u_left_bd_trial = fem.make_trial(space=self._u_space, domain=self._left)
        u_left_bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": u_left_bd_trial, "v": u_left_bd_test},
            assembly="nodal",
            output_dtype=float,
        )
        fem.normalize_dirichlet_projector(u_left_bd_matrix)
        self._bd_projector = u_left_bd_matrix
        
        self.E_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=float)

        self._nu = poisson_ratio
        self._load = wp.array([load[0], load[1], load[2]], dtype=float, requires_grad=True)
        self._load.requires_grad=True

        self._u_right_test = fem.make_test(space=self._u_space, domain=self._right)


        # forward ################################################################################################

        u = self._u_field_meas.dof_values
        u.zero_()

        u_rhs = wp.empty(self._u_space.node_count(), dtype=wp.vec3f, requires_grad=True)

        E_space_meas = fem.make_polynomial_space(self._geo, degree=degree, dtype=float)
        self._E_field_meas = fem.make_discrete_field(space=E_space_meas)

        E_meas_init = np.zeros((E_space_meas.node_count()))*1.0e7+25.0e9
        damage_idx_start = int(E_space_meas.node_count()*0.5) - resolution[1]//2
        damage_idx_width = int((resolution[1]+1)*2)
        damage_idx = np.arange(damage_idx_start, damage_idx_start+damage_idx_width)
        E_meas_init[damage_idx] = 24.0e9
        
        self._E_field_meas.dof_values = wp.array(E_meas_init, dtype=float, requires_grad=True)
        self._E_field_meas.dof_values.requires_grad = True

        fem.integrate(
            applied_load_form,
            fields={"v": self._u_right_test},
            values={"load": self._load},
            output=u_rhs,
        )

        u_matrix_meas = fem.integrate(
            hooke_elasticity_form,
            fields={"u": self._u_trial, "v": self._u_test, "E_field": self._E_field_meas},
            values={"nu": self._nu},
            output_dtype=float,
        )
        fem.project_linear_system(u_matrix_meas, u_rhs, self._bd_projector, normalize_projector=False)
        fem_example_utils.bsr_cg(u_matrix_meas, b=u_rhs, x=u, quiet=self._quiet, tol=1e-6, max_iters=1000)
        
        self.strain_space_meas = fem.make_polynomial_space(
            self._geo,
            degree=1,
            dtype=wp.mat33,   # tensor type
        )
        self.strain_field_meas = self.strain_space_meas.make_field()
        fem.interpolate(
                strain_field,
                dest=self.strain_field_meas,
                fields={"u": self._u_field_meas},
            )
        

    # backward ################################################################################################
        # Initialize Adam optimizer
        # Current implementation assumes scalar arrays, so cast our vec2 arrays to scalars
        N = self.E_space.node_count()
        self.init_E = np.zeros(N)+30.0e9
        self.E_array = wp.array(self.init_E, dtype=float, requires_grad=True)
        self.params = wp.array(self.E_array, dtype=wp.float32).flatten()
        self.params.grad = wp.array(self.E_array.grad, dtype=wp.float32).flatten()
        self.optimizer = Adam([self.params], lr=self.lr)

    def step(self):
        self.tape = wp.Tape()

        self._E_field = fem.make_discrete_field(space=self.E_space)
        self._E_field.dof_values = self.E_array
        self._E_field.dof_values.requires_grad = True

        # Forward step, record adjoint self.tape for forces
        u_est = self._u_field.dof_values
        u_est.zero_()

        u_rhs = wp.empty(self._u_space.node_count(), dtype=wp.vec3f, requires_grad=True)

        with self.tape:
            fem.integrate(
                applied_load_form,
                fields={"v": self._u_right_test},
                values={"load": self._load},
                output=u_rhs,
            )
            # the elastic force will be zero at the first iteration,
            # but including it on the tape is necessary to compute the gradient of the force equilibrium
            # using the implicit function theorem
            # Note that this will be evaluated in the backward pass using the updated values for "_u_field"
            fem.integrate(
                hooke_elasticity_form,
                fields={"u": self._u_field, "v": self._u_test, "E_field": self._E_field},
                values={"nu": -self._nu},
                output=u_rhs,
                add=True,
            )

        u_matrix = fem.integrate(
            hooke_elasticity_form,
            fields={"u": self._u_trial, "v": self._u_test, "E_field": self._E_field},
            values={"nu": self._nu},
            output_dtype=float,
        )

        fem.project_linear_system(u_matrix, u_rhs, self._bd_projector, normalize_projector=False)
        fem_example_utils.bsr_cg(u_matrix, b=u_rhs, x=u_est, quiet=self._quiet, tol=1e-6, max_iters=1000)

        # Record adjoint of linear solve
        # (For nonlinear elasticity, this should use the final hessian, as per implicit function theorem)
        def solve_linear_system():
            fem_example_utils.bsr_cg(u_matrix, b=u_est.grad, x=u_rhs.grad, quiet=self._quiet, tol=1e-6, max_iters=1000)
            u_rhs.grad -= self._bd_projector @ u_rhs.grad
            self._u_field.dof_values.grad.zero_()

        self.tape.record_func(solve_linear_system, arrays=(u_rhs, u_est))

        # Evaluate residual
        # Integral of squared difference between simulated position and target positions
        loss = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)

        with self.tape:
            fem.integrate(
                loss_form,
                # loss_disp,
                fields={"u": self._u_field, "u_meas": self._u_field_meas},
                domain=self._u_test.domain,
                output=loss,
            )

        # perform backward step
        self.tape.backward(loss=loss)

        # update positions and reset self.tape
        self.optimizer.step([-self.params.grad])
        grad = -self.E_array.grad.numpy()
        self.tape.zero()
        

        self.strain_field = self.strain_space_meas.make_field()
        fem.interpolate(
                strain_field,
                dest=self.strain_field,
                fields={"u": self._u_field},
            )

        return loss.numpy(), self.E_array.numpy()[0]


with wp.ScopedDevice(None):
    resolution = (200, 12, 12)
    example = Example(
        quiet=True,
        degree=1,
        resolution=(200, 12, 12),
        mesh="quad",
        poisson_ratio=0.3,
        load=wp.vec3(2.0e5*10.0, 0.0, 0.0),
        lr=5.0e8,
    )

    losses = []
    params = []
    n_its = 50
    from tqdm import tqdm
    for _ in tqdm(np.arange(n_its)):
        loss, param = example.step()
        losses.append(loss)
        params.append(param)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

ax = axes[0]
ax.plot(np.arange(n_its), losses)
# ax.hlines(25e9, 0, 1e1, color='r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title('Adam Loss Evolution')
ax.set_yscale('log')
ax.set_ylim([np.min(losses), np.max(losses)])

ax = axes[1]
ax.plot(np.arange(n_its), params)
ax.hlines(25e9, 0, n_its, color='r', alpha=0.4)
ax.set_xlabel('Iterations')
ax.set_ylabel('E')
ax.set_title('Adam Learning Curve')

plt.savefig("adam_nh.png", dpi=300)
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

deformation_scale=10000
# Get data
node_positions = example._u_space.node_positions().numpy()
disp_meas = example._u_field_meas.dof_values.numpy()
disp_est = example._u_field.dof_values.numpy()
strain_meas = example.strain_field_meas.dof_values.numpy()
strain_est = example.strain_field.dof_values.numpy()

node_positions = example._u_space.node_positions().numpy()
E_meas = example._E_field_meas.dof_values.numpy()
E_est = example._E_field.dof_values.numpy()

disp_min = np.min((np.min(disp_meas[:,0]), np.min(disp_est[:,0])))
disp_max = np.max((np.max(disp_meas[:,0]), np.max(disp_est[:,0])))
strain_min = np.min((np.min(strain_meas[:,0]), np.min(strain_est[:,0])))
strain_max = np.max((np.max(strain_meas[:,0]), np.max(strain_est[:,0])))
E_min = np.min((np.min(E_meas), np.min(E_est), np.min(example.init_E)))
E_max = np.max((np.max(E_meas), np.max(E_est), np.max(example.init_E)))


# 1.
# Deformed positions
deformed_pos = node_positions + deformation_scale * disp_meas
ax = axes[0]
x = node_positions[:, 0]
y = node_positions[:, 1]
ax.scatter(x, y, c='blue', s=1, alpha=0.3, label='Original')
# Deformed shape
x_def = deformed_pos[:, 0]
y_def = deformed_pos[:, 1]
ax.scatter(x_def, y_def, c='red', s=1, alpha=0.5, label=f'Deformed (×{deformation_scale})')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('True Deformed Shape', fontweight='bold')
# ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 2. Elastic fields
ax = axes[1]
scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1],
                    c=E_meas, cmap='jet', s=10, vmin=E_min, vmax=E_max)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Young\'s Modulus (Pa)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('True Elastic Field', fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax = axes[2]
scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1],
                    c=example.init_E, cmap='jet', s=10, vmin=E_min, vmax=E_max)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Young\'s Modulus (Pa)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Initial Elastic Field', fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax = axes[3]
scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1],
                    c=E_est, cmap='jet', s=10, vmin=E_min, vmax=E_max)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Young\'s Modulus (Pa)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Estimated Elastic Field', fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)


plt.savefig("adam_nh_compare.png", dpi=300)
plt.show()
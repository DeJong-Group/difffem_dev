import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.optim import Adam

import json

@wp.kernel
def clip_params_kernel(
    params: wp.array(dtype=float),
    concrete_indices: wp.array(dtype=int),
    rebar_indices: wp.array(dtype=int),
    cell_slice_map_concrete: wp.array(dtype=int),
    cell_slice_map_rebar: wp.array(dtype=int),
    full_E_field: wp.array(dtype=float),
    n_slices: int,
    n_concrete: int,
    n_rebar: int,
    freeze_rebar: bool,
    rebar_static_val: float,
    c_min: float,
    c_max: float,
    r_min: float,
    r_max: float
):
    tid = wp.tid()
    
    if opt_slice:
        if tid < n_concrete:
            c_idx = concrete_indices[tid]
            s_idx = cell_slice_map_concrete[tid]
            # Concrete parameters are in the first track: [0 : n_slices]
            val = params[s_idx]
            if val < c_min:
                params[s_idx] = c_min
            elif val > c_max:
                params[s_idx] = c_max
            full_E_field[c_idx] = params[s_idx]
            
        # Map Rebar Slices
        if tid < n_rebar:
            r_idx = rebar_indices[tid]
            if freeze_rebar:
                full_E_field[r_idx] = rebar_static_val
            else:
                # s_idx = cell_slice_map_rebar[tid]
                # Rebar parameters are in the second track: [n_slices : 2*n_slices]
                val = params[n_slices]
                if val < r_min:
                    params[n_slices] = r_min
                elif val > r_max:
                    params[n_slices] = r_max
                full_E_field[r_idx] = params[n_slices]
    else:
        if tid < n_concrete:
            c_idx = concrete_indices[tid]
            s_idx = cell_slice_map_concrete[tid]
            # Concrete parameters are in the first track: [0 : n_slices]
            val = params[s_idx]
            if val < c_min:
                params[s_idx] = c_min
            elif val > c_max:
                params[s_idx] = c_max
            full_E_field[c_idx] = params[s_idx]

        # Map Rebar Slices
        if tid < n_rebar:
            r_idx = rebar_indices[tid]
            if freeze_rebar:
                full_E_field[r_idx] = rebar_static_val
            else:
                # s_idx = cell_slice_map_rebar[tid]
                # Rebar parameters are in the second track: [n_slices : 2*n_slices]
                if val < r_min:
                    params[n_concrete] = r_min
                elif val > r_max:
                    params[n_concrete] = r_max
                full_E_field[r_idx] = params[n_concrete]


@wp.kernel
def map_slices_to_cells2(
    params: wp.array(dtype=float),
    concrete_indices: wp.array(dtype=int),
    rebar_indices: wp.array(dtype=int),
    cell_slice_map_concrete: wp.array(dtype=int),
    cell_slice_map_rebar: wp.array(dtype=int),
    full_E_field: wp.array(dtype=float),
    n_slices: int,
    n_concrete: int,
    n_rebar: int,
    freeze_rebar: bool,
    rebar_static_val: float
):
    tid = wp.tid()
    
    # Map Concrete Slices
    if tid < n_concrete:
        c_idx = concrete_indices[tid]
        s_idx = cell_slice_map_concrete[tid]
        # Concrete parameters are in the first track: [0 : n_slices]
        full_E_field[c_idx] = params[s_idx]
        
    # Map Rebar Slices
    if tid < n_rebar:
        r_idx = rebar_indices[tid]
        if freeze_rebar:
            full_E_field[r_idx] = rebar_static_val
        else:
            # s_idx = cell_slice_map_rebar[tid]
            # Rebar parameters are in the second track: [n_slices : 2*n_slices]
            full_E_field[r_idx] = params[n_slices]# + s_idx]

@wp.kernel
def map_rebar(
    params: wp.array(dtype=float),
    concrete_indices: wp.array(dtype=int),
    rebar_indices: wp.array(dtype=int),
    cell_slice_map_concrete: wp.array(dtype=int),
    cell_slice_map_rebar: wp.array(dtype=int),
    full_E_field: wp.array(dtype=float),
    n_slices: int,
    n_concrete: int,
    n_rebar: int,
    freeze_rebar: bool,
    rebar_static_val: float,
):
    tid = wp.tid()
    # Map Concrete Slices
    if tid < n_concrete:
        c_idx = concrete_indices[tid]
        s_idx = cell_slice_map_concrete[tid]
        # Concrete parameters are in the first track: [0 : n_slices]
        full_E_field[c_idx] = params[s_idx]
    # Map Rebar Slices
    if tid < n_rebar:
        r_idx = rebar_indices[tid]
        if freeze_rebar:
            full_E_field[r_idx] = rebar_static_val
        else:
            # s_idx = cell_slice_map_rebar[tid]
            # Rebar parameters are in the second track: [n_slices : 2*n_slices]
            full_E_field[r_idx] = params[n_concrete]

@wp.kernel
def map_slices_to_cells(
    slice_values: wp.array(dtype=float),
    cell_slice_map: wp.array(dtype=int),
    concrete_indices: wp.array(dtype=int),
    n_concrete: int,
    full_E_field: wp.array(dtype=float),
    rebar_E: float,
    rebar_indices: wp.array(dtype=int)
):
    tid = wp.tid()
    # Map concrete
    if tid < n_concrete:
        c_idx = concrete_indices[tid]
        s_idx = cell_slice_map[tid]
        full_E_field[c_idx] = slice_values[s_idx]
    

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
    # face_ab: wp.array(dtype=wp.float32),
    face_ab: wp.array(dtype=int),
    x_bound: wp.float32,
    z_bound: wp.float32,
):
    nor = fem.normal(domain, s)
    pos = fem.position(domain, s)

    # if nor[0] < -0.5:
    #     left[s.qp_index] = 1
    # elif nor[0] > 0.5:
    #     right[s.qp_index] = 1
    if (pos[1]>=0.0487 and pos[1]<=0.0713) and (pos[2]>=0.0487 and pos[2]<=0.0713):
        if pos[0] == 0.0:
            left[s.qp_index] = 1
        elif pos[0] == x_bound:
            right[s.qp_index] = 1
    # elif pos[2] == 0.0 or pos[2] == z_bound:
    #     face_ab[s.qp_index] = 1
    if ((pos[2]>=0.0487 and pos[2]<=0.0713)) and (pos[1]>=0.0487 and pos[1]<=0.0713):
        face_ab[s.qp_index] = 1
    # face_ab[s.qp_index] = 1
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
def hooke_elasticity_form(
    s: fem.Sample, 
    u: fem.Field, 
    v: fem.Field, 
    E_field: fem.Field, 
    nu: float,
):
    E_val = E_field(s)
    l = lame_from_E_nu(E_val, nu)
    lamb = l[0]
    mu = l[1]
    stress = hooke_stress(fem.D(u, s), lamb, mu)
    return wp.ddot(fem.D(v, s), stress)

@fem.integrand
def applied_load_form(
    s: fem.Sample, 
    domain: fem.Domain, 
    v: fem.Field, 
    load: wp.array(dtype=wp.float32)
):
    return v(s)[0]*load[0] + v(s)[1]*load[1] + v(s)[2]*load[2]

@fem.integrand
def loss_disp(
    s: fem.Sample, 
    domain: fem.Domain, 
    u: fem.Field, 
    u_meas: fem.Field
):
    disp = u(s)
    disp_meas = u_meas(s)
    diff = disp - disp_meas
    stress_norm_sq = 0.5 * wp.dot(diff, diff)

    return stress_norm_sq 

@fem.integrand
def loss_form(
    s: fem.Sample, 
    domain: fem.Domain, 
    u: fem.Field,
    u_meas: fem.Field, 
    mask: wp.array(dtype=int),
    strain_meas_arr: wp.array(dtype=float),
    strain_est: wp.array(dtype=float),
    sensor_positions: wp.array(dtype=wp.vec3f),
    weight: float,
):
    strain = strain_field(s, u)[0,0]
    strain_meas = strain_meas_arr[s.qp_index]
    strain_est[s.qp_index] = strain
    diff = (strain - strain_meas)
    axial = 0.5 * (diff ** 2.0)# + (diff[1,1] ** 2.0) + (diff[2,2] ** 2.0))
    # stress_norm_sq = 0.5 * wp.ddot(diff, diff) * 1e20
    stress_norm_sq = axial * 1e20
    return stress_norm_sq * weight 

@fem.integrand
def loss_form_DIC(
    s: fem.Sample, 
    domain: fem.Domain, 
    u: fem.Field,
    target_strain: wp.array(dtype=float),
    weight: float,
):
    # Extract xx component of strain: [0,0]
    strain = fem.D(u, s)[0,0]
    strain_meas = target_strain[s.qp_index]
    
    diff = strain - strain_meas
    return 0.5 * (diff * diff) * 1e20 * weight


class Example:
    def __init__(
        self,
        quiet=False,
        degree=1,
        resolution=(200, 12, 12),
        mesh="tri",
        poisson_ratio=0.3,
        E=25.0e9,
        c_lim=(1000.0, 25.0e9*1.1),
        r_lim=(200.0e9*0.9, 200.0e9*1.1),
        load=(1.0, 0.0, 0.0),
        lr=1.0e-3,
        strain_meas = None,
        strain_meas_x = None,
        Exx = (None, None),
        A_pos = (None, None),
        B_pos = (None, None),
        loss_weights = [1.0, 1.0, 1.0],
        u_meas = None,
        freeze_rebar = False,
        opt_slice = False,
    ):
        self._quiet = quiet
        self.degree = degree
        self.lr = lr
        # procedural rectangular domain definition
        bounds_lo = wp.vec3(0.0, 0.0, 0.0)
        bounds_hi = wp.vec3(1.0, 0.12, 0.12)
        self._initial_volume = (bounds_hi - bounds_lo)[0] * (bounds_hi - bounds_lo)[1] * (bounds_hi - bounds_lo)[2]
        

        # optimization settings
        self.freeze_rebar = freeze_rebar
        self.opt_slice = opt_slice

        self.u_meas = u_meas

        self.resolution = resolution

        # number of nodes
        node_x = np.linspace(bounds_lo[0], bounds_hi[0], self.resolution[0]+1)
        # node_y = np.linspace(bounds_lo[1], bounds_hi[1], self.resolution[1])
        # node_z = np.linspace(bounds_lo[2], bounds_hi[2], self.resolution[2])
        node_y = np.array([bounds_lo[1], 0.025, 0.0487, 0.0713, 0.095, bounds_hi[1]])
        node_z = np.array([bounds_lo[2], 0.025, 0.0487, 0.0713, 0.095, bounds_hi[2]])

        # number of elements/cells    
        self.Nx = self.resolution[0]
        self.Ny = len(node_y) - 1
        self.Nz = len(node_z) - 1

        positions_np = np.transpose(np.meshgrid(node_x, node_y, node_z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)
        positions = wp.array(positions_np, dtype=wp.vec3)
        
        target_vals = np.array([0.0487, 0.0713])
        tol = 1e-6
        y_mask = np.isclose(positions_np[:, 1][:, None], target_vals, atol=tol).any(axis=1)
        z_mask = np.isclose(positions_np[:, 2][:, None], target_vals, atol=tol).any(axis=1)

        # self.rebar_indices = np.where(y_mask & z_mask)[0]
        n_sensor = len(strain_meas)
        extended_strain_np = np.broadcast_to(strain_meas[:, None, None], (n_sensor, 2, 2))
        extended_strain_np_reshape = extended_strain_np.reshape(-1)
        self.strain_meas = wp.from_numpy(extended_strain_np_reshape, dtype=float, requires_grad=True)
        self.strain_est = wp.empty(extended_strain_np_reshape.shape, dtype=float, requires_grad=True)
        self.positions_sensors_np = np.transpose(np.meshgrid(strain_meas_x, target_vals, target_vals, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)
        self.positions_sensors = wp.array(self.positions_sensors_np, dtype=wp.vec3, requires_grad=True)

        self.c_lim = c_lim
        self.r_lim = r_lim
        # --- DIC Data Processing ---
        def prep_dic(x_grid, y_grid, exx_data, z_val):
            pts = np.stack([x_grid.flatten(), y_grid.flatten(), np.full_like(x_grid.flatten(), z_val)], axis=-1)
            return wp.array(pts, dtype=wp.vec3), wp.array(exx_data.flatten(), dtype=float)

        self.strain_A, self.strain_B = Exx
        A_x, A_y = A_pos
        B_x, B_y = B_pos

        # Side A (Z=0)
        self.pos_A, self.strain_A = prep_dic(A_x, A_y, Exx_A, 0.00)
        # Side B (Z=0.12)
        self.pos_B, self.strain_B = prep_dic(B_x, B_y, Exx_B, 0.12)
    

        # Loss control weights (Set these via input or class args)
        self.w_rebar = loss_weights[0]
        self.w_sideA = loss_weights[1]
        self.w_sideB = loss_weights[2]

        # print(extended_strain_np_reshape.shape, positions_sensors_np.shape)
        # sample = np.arange(100)
        # for i in sample:
        #     print(i, positions_sensors_np[i], extended_strain_np_reshape[i])
        # print(positions_sensors_np)
        # print(extended_strain_np_reshape)
        if mesh == "tri":
            # triangle mesh, optimize vertices directly
            positions, tri_vidx = fem_example_utils.gen_tetmesh(
                res=wp.vec3i(resolution[0], resolution[1], resolution[2]), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            self._geo = fem.Tetmesh(tet_vertex_indices=tri_vidx, positions=positions, build_bvh=True)
            self._vertex_positions = positions
        elif mesh == "quad":
            vidx = fem.utils.grid_to_hexes(self.Nx, self.Ny, self.Nz)
            quad_vidx = wp.array(vidx, dtype=int)
            self._geo = fem.Hexmesh(hex_vertex_indices=quad_vidx, positions=positions, build_bvh=True)
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

        self.rebar_indices = np.arange(12,self._geo.cell_count(), 25)
        concrete_indices = np.delete(np.arange(self._geo.cell_count()), self.rebar_indices)

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
        self.cells = fem.Cells(self._geo)
        left_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        right_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        # self.ab_mask = wp.zeros(shape=boundary.element_count(), dtype=float)
        self.ab_mask = wp.zeros(shape=self.cells.element_count(), dtype=int)
        # print("cells: ", boundary.element_count(), self.cells.element_count())
        
        fem.interpolate(
            classify_boundary_sides,
            quadrature=fem.RegularQuadrature(boundary, order=0),
            values={"left": left_mask, 
                    "right": right_mask, 
                    "face_ab": self.ab_mask,
                    "x_bound": bounds_hi[0],
                    "z_bound": bounds_hi[2]
                    },
        )

        self._left = fem.Subdomain(boundary, element_mask=left_mask)
        self._right = fem.Subdomain(boundary, element_mask=right_mask)
        self._face = fem.Subdomain(boundary, element_mask=self.ab_mask)
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
        

        self._nu = poisson_ratio
        self.load = load
        self._load = wp.array([load[0], load[1], load[2]], dtype=float, requires_grad=True)
        self._load.requires_grad=True

        self._u_right_test = fem.make_test(space=self._u_space, domain=self._right)


        # forward ################################################################################################

        u = self._u_field_meas.dof_values
        u.zero_()

        u_rhs = wp.empty(self._u_space.node_count(), dtype=wp.vec3f, requires_grad=True)

        E_space_meas = fem.make_polynomial_space(self._geo, degree=0, dtype=float)
        self._E_field_meas = fem.make_discrete_field(space=E_space_meas)
        E_meas_init = np.zeros((E_space_meas.node_count()))*1.0e7+25.0e9
        

        damage_idx_start = int(E_space_meas.node_count()*0.5) #- ((self.Ny)*(self.Nz))//2
        damage_idx_width = int((self.Ny)*(self.Nz))
        # damage_idx_start = int(E_space_meas.node_count()*0.5)# - ((resolution[1]+1)*(resolution[2]+1))//2
        # damage_idx_width = int((resolution[1]+1)*(resolution[2]+1))
        damage_idx = np.arange(damage_idx_start, damage_idx_start+damage_idx_width)
        E_meas_init[damage_idx] = 25.0e9*0.9
        E_meas_init[self.rebar_indices] = 200.0e9


        
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
        # print(self.strain_field_meas.dof_values.numpy().shape)
        fem.interpolate(
                strain_field,
                dest=self.strain_field_meas.dof_values,
                fields={"u": self._u_field_meas},
                at=fem.NodalQuadrature(fem.Cells(self._geo), self._u_space),
                # dest_space=self.strain_space_meas,
            )
        

    # backward ################################################################################################
        # Initialize Adam optimizer
        # Current implementation assumes scalar arrays, so cast our vec2 arrays to scalars
        self.E_space = fem.make_polynomial_space(self._geo, degree=0, dtype=float)
        N = self.E_space.node_count()
        self.init_E = np.zeros(N)+25.0e9
        self.init_E[self.rebar_indices] = 200.0e9
        self.E_array = wp.array(self.init_E, dtype=float, requires_grad=True)
        self.E_hist = []



        if not self.opt_slice:
            all_cells = np.arange(self._geo.cell_count())
            self.concrete_indices = np.delete(all_cells, self.rebar_indices)
            self.n_concrete = len(self.concrete_indices)
            self.cell_slice_map_concrete = wp.array(np.arange(self.n_concrete), dtype=int)
            self.cell_slice_map_rebar = wp.array((self.rebar_indices // (self.Ny * self.Nz)).astype(int), dtype=int)
            self.n_concrete = len(self.concrete_indices)
            self.n_rebar = len(self.rebar_indices)
            if self.freeze_rebar:
                self.params = wp.array(self.E_array, dtype=wp.float32).flatten()
                self.params.grad = wp.array(self.E_array.grad, dtype=wp.float32).flatten()
            else:
                all_cells = np.arange(self._geo.cell_count())
                self.concrete_indices = np.delete(all_cells, self.rebar_indices)
                self.n_concrete = len(self.concrete_indices)
                self.n_rebar = len(self.rebar_indices)
                self.cell_slice_map_concrete = wp.array(np.arange(self.n_concrete), dtype=int)
                self.cell_slice_map_rebar = wp.array((self.rebar_indices // (self.Ny * self.Nz)).astype(int), dtype=int)
                
                c_init = np.full(self.n_concrete, 25.0e9, dtype=np.float32)
                r_init = np.array([200.0e9], dtype=np.float32)
                init_params = np.concatenate([c_init, r_init])
                self.params = wp.array(init_params, dtype=wp.float32, requires_grad=True)
        else:
            self.num_slices = self.Nx 
            all_cells = np.arange(self._geo.cell_count())
            self.concrete_indices = np.delete(all_cells, self.rebar_indices)
            self.cell_slice_map_concrete = wp.array((self.concrete_indices // (self.Ny * self.Nz)).astype(int), dtype=int)
            self.cell_slice_map_rebar = wp.array((self.rebar_indices // (self.Ny * self.Nz)).astype(int), dtype=int)
            self.n_concrete = len(self.concrete_indices)
            self.n_rebar = len(self.rebar_indices)
            # self.cell_slice_map = (self.concrete_indices // (self.Ny * self.Nz)).astype(int)
            # self.slice_E_init = np.full(self.num_slices, 25.0e9, dtype=np.float32)
            # self.params = wp.array(self.slice_E_init, dtype=wp.float32, requires_grad=True)
            if self.freeze_rebar:
                # Only Concrete Slices
                init_params = np.full(self.num_slices, 25.0e9, dtype=np.float32)
            else:
                # Concatenate Concrete slices and Rebar slices
                c_slices = np.full(self.num_slices, 25.0e9, dtype=np.float32)
                r_slices = np.array([200.0e9], dtype=np.float32)
                init_params = np.concatenate([c_slices, r_slices])
            
            self.params = wp.array(init_params, dtype=wp.float32, requires_grad=True)
        self.optimizer = Adam([self.params], lr=self.lr)

    def step(self):
        self.tape = wp.Tape()
        if self.opt_slice:
            with self.tape:
                wp.launch(
                    kernel=map_slices_to_cells2,
                    dim=max(self.n_concrete, self.n_rebar), # Launch for the larger set
                    inputs=[
                        self.params, 
                        wp.array(self.concrete_indices, dtype=int),
                        wp.array(self.rebar_indices, dtype=int),
                        self.cell_slice_map_concrete,
                        self.cell_slice_map_rebar,
                        self.E_array,
                        self.Nx,        # n_slices
                        self.n_concrete,
                        self.n_rebar,
                        self.freeze_rebar,
                        200.0e9         # rebar_static_val
                    ]
                )
        else:
            if not self.freeze_rebar:
                with self.tape:
                    wp.launch(
                        kernel=map_rebar,
                        dim=max(self.n_concrete, self.n_rebar), # Launch for the larger set
                        inputs=[
                            self.params, 
                            wp.array(self.concrete_indices, dtype=int),
                            wp.array(self.rebar_indices, dtype=int),
                            self.cell_slice_map_concrete,
                            self.cell_slice_map_rebar,
                            self.E_array,
                            self.Nx,        # n_slices
                            self.n_concrete,
                            self.n_rebar,
                            self.freeze_rebar,
                            200.0e9,         # rebar_static_val
                        ]
                    )

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
        loss = wp.zeros(shape=1, dtype=wp.float32, requires_grad=True)
        with self.tape:
            if self.w_rebar > 0.0:
                fem.integrate(
                    loss_form,
                    # loss_disp,
                    fields={"u": self._u_field, "u_meas": self._u_field_meas},
                    values={
                        "mask": self.ab_mask, 
                        "strain_meas_arr": self.strain_meas, 
                        "strain_est": self.strain_est,
                        "sensor_positions": self.positions_sensors,
                        "weight": self.w_rebar,
                        },
                    domain=self.cells,
                    quadrature=fem.PicQuadrature(self.cells, self.positions_sensors),
                    output=loss,
                    add=True
                )
            if self.w_sideA > 0.0:
                fem.integrate(
                    loss_form_DIC,
                    fields={"u": self._u_field},
                    values={"target_strain": self.strain_A, "weight": self.w_sideA},
                    domain=self.cells,
                    quadrature=fem.PicQuadrature(self.cells, self.pos_A),
                    output=loss,
                    add=True
                )

            # 3. Side B DIC Loss
            if self.w_sideB > 0.0:
                fem.integrate(
                    loss_form_DIC,
                    fields={"u": self._u_field},
                    values={"target_strain": self.strain_B, "weight": self.w_sideB},
                    domain=self.cells,
                    quadrature=fem.PicQuadrature(self.cells, self.pos_B),
                    output=loss,
                    add=True
                )

        # perform backward step
        self.tape.backward(loss=loss)

        # update positions and reset self.tape
        if not self.opt_slice:
            if self.freeze_rebar:
                grad = -self.E_array.grad.numpy()
                grad[self.rebar_indices] = 0.0
                # self.optimizer.step([wp.array(grad, dtype=wp.float32)])
            else:
                grad = -self.params.grad
            self.optimizer.step([wp.array(grad, dtype=wp.float32)])
        # self.optimizer.step([-self.params.grad])
        else:
            grad = -self.params.grad
            self.optimizer.step([wp.array(grad, dtype=wp.float32)])

        wp.launch(
            kernel=clip_params_kernel,
            dim=max(self.n_concrete, self.n_rebar),
            inputs=[
                self.params, 
                wp.array(self.concrete_indices, dtype=int),
                wp.array(self.rebar_indices, dtype=int),
                self.cell_slice_map_concrete,
                self.cell_slice_map_rebar,
                self.E_array,
                self.Nx,        # n_slices
                self.n_concrete,
                self.n_rebar,
                self.freeze_rebar,
                200.0e9,         # rebar_static_val
                self.c_lim[0], self.c_lim[1], 
                self.r_lim[0], self.r_lim[1]
            ]
        )

        self.tape.zero()
        # print(loss.numpy(), self.params.numpy().min(), self.params.numpy().max())#.tolist(), grad)
        # print(loss.numpy(), self.params.numpy().min(), self.params.numpy().max())
        # print(grad)
        self.strain_field = self.strain_space_meas.make_field()
        fem.interpolate(
                strain_field,
                dest=self.strain_field,
                fields={"u": self._u_field},
                # at=fem.NodalQuadrature(fem.Cells(self._geo), self._u_space)
            )
        self.E_hist.append(self._E_field.dof_values.numpy().tolist())
        return loss.numpy().tolist(), self.E_array.numpy().tolist(), self.strain_est.numpy().tolist()

import argparse
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("specimen", type=str)
parser.add_argument("loadstep", type=str)
parser.add_argument("freeze_rebar", type=str)
parser.add_argument("opt_slice", type=str)
parser.add_argument("lr", type=str)
parser.add_argument("w_rebar", type=str)
parser.add_argument("w_A", type=str)
parser.add_argument("w_B", type=str)
parser.add_argument("step", type=str)
parser.add_argument("data", type=str)

args = parser.parse_args()
print(args)
specimen = args.specimen
loadstep = args.loadstep
freeze_rebar = bool(int(args.freeze_rebar))
opt_slice = bool(int(args.opt_slice))
lr = int(float(args.lr))
weights = [
    float(args.w_rebar),
    float(args.w_A),
    float(args.w_B),
]
step = int(float(args.step))
if args.data == '':
    data = 'rcrt_adjust'
else:
    data = args.data

# freeze_rebar = False
# opt_slice = True
# specimen = "1"
# loadstep = "1"
gauge_pitch = 0.0013 # mm
# load target
# strain_meas_l = np.load(f'rebar_strains/s{specimen}_ls{loadstep}_l.npy', allow_pickle=True)
# strain_meas_r = np.load(f'rebar_strains/s{specimen}_ls{loadstep}_r.npy', allow_pickle=True)
# strain_meas = (strain_meas_l + strain_meas_r) *1e-6 / 2 

print('loading target')
strain_meas = np.load(f'rebar_strains/s{specimen}_ls{loadstep}_combined.npy', allow_pickle=True) * 1e-6
strain_meas_x = np.arange(len(strain_meas)) * gauge_pitch

if loadstep=='1':
    kips = '5'
Exx_A = np.load(f'DIC/S{specimen}A_{kips}k_Exx_{data}.npy')
Exx_B = np.load(f'DIC/S{specimen}B_{kips}k_Exx_{data}.npy')

A_x = np.linspace(0,1,Exx_A.shape[1], endpoint=True)
A_y = np.linspace(0,0.12,Exx_A.shape[0], endpoint=True)
A_x, A_y = np.meshgrid(A_x, A_y)

B_x = np.linspace(0,1,Exx_B.shape[1], endpoint=True)
B_y = np.linspace(0,0.12,Exx_B.shape[0], endpoint=True)
B_x, B_y = np.meshgrid(B_x, B_y)
print(Exx_A.shape, Exx_B.shape, strain_meas_x.shape)
# calculate load
if specimen in ['1', '2', '3']:
    loads = np.array([22.2, 44.5, 89.0, 89.0, 133.5, 177.9]) * 1000
else:
    loads = np.array([22.2, 44.5, 89.0, 133.5, 177.9, 222.4]) * 1000
load = loads[int(loadstep) - 1]

prev_loss = None
ftol = 1e-16  # Tolerance for relative change


print('initializing Warp')
with wp.ScopedDevice("cuda:0"):
    
    example = Example(
        quiet=True,
        degree=1,
        resolution=(100, 6, 6),
        mesh="quad",
        poisson_ratio=0.3,
        load=wp.vec3(load/0.000509, 0.0, 0.0),
        lr=lr,
        strain_meas = strain_meas,
        strain_meas_x = strain_meas_x,
        Exx = (Exx_A, Exx_B),
        A_pos = (A_x, A_y),
        B_pos = (B_x, B_y),
        loss_weights = weights,
        freeze_rebar = freeze_rebar,
        opt_slice = opt_slice,
    )

    losses = []
    params = []
    strains = []
    n_its = step
    from tqdm import tqdm
    from collections import deque

    # Settings
    window_size = 100
    loss_window = deque(maxlen=window_size)
    prev_avg_loss = None

    # for i in tqdm(range(n_its)):
    for i in range(n_its):
        loss, param, strain = example.step()
        
        # Store current loss (assuming loss is a list/array [val])
        current_loss_val = loss[0]
        losses.append(current_loss_val)
        params.append(param)
        strains.append(strain)

        # Add to rolling window
        loss_window.append(current_loss_val)

        # Only check convergence once the window is full
        if len(loss_window) == window_size:
            current_avg_loss = np.mean(loss_window)

            if prev_avg_loss is not None:
                # Convergence Formula: (f_avg^k - f_avg^{k+1}) / max(|f_avg^k|, |f_avg^{k+1}|, 1e-6)
                diff = np.abs(prev_avg_loss - current_avg_loss)
                denom = np.max([np.abs(prev_avg_loss), np.abs(current_avg_loss), 1e-6])
                rel_change = diff / denom
                
                if rel_change <= ftol:
                    print(f"\nConverged (Moving Average) at iteration {i}: "
                        f"Rel Change {rel_change:.2e} <= {ftol}")
                    n_its = i + 1
                    break
            
            # Update the reference average for the next iteration
            prev_avg_loss = current_avg_loss

    # Ensure n_its is updated if we finish the loop without breaking
    else:
        n_its = len(losses)




# Get data
node_positions = example._u_space.node_positions().numpy()
disp_meas = example._u_field_meas.dof_values.numpy()
disp_est = example._u_field.dof_values.numpy()
# strain_meas = example.strain_field_meas.dof_values.numpy()
# strain_est = example.strain_field.dof_values.numpy()

best_index = np.argmin(losses)
E_best = params[best_index]
strain_best = strains[best_index]

node_positions = example._u_space.node_positions().numpy()
E_meas = example._E_field_meas.dof_values.numpy()
E_est = example._E_field.dof_values.numpy()

disp_min = np.min((np.min(disp_meas[:,0]), np.min(disp_est[:,0])))
disp_max = np.max((np.max(disp_meas[:,0]), np.max(disp_est[:,0])))
# strain_min = np.min((np.min(strain_meas[:,0,0]), np.min(strain_est[:,0,0])))
# strain_max = np.max((np.max(strain_meas[:,0,0]), np.max(strain_est[:,0,0])))
E_min = np.min((np.min(E_meas), np.min(E_est), np.min(example.init_E), np.min(E_best)))
E_max = np.max((np.max(E_meas), np.max(E_est), np.max(example.init_E)))

node_positions = example._u_space.node_positions().numpy()
disp_meas = example._u_field_meas.dof_values.numpy()
strain_res  = example.strain_field.dof_values.numpy()

node_positions = example._u_space.node_positions().numpy()
E_meas = example._E_field_meas.dof_values.numpy()

P = load
A = 0.12*0.12
E = np.mean(E_meas)
L = 1.0
dx = L / 100
theoretical_strain = P / (A * E)
theoretical_displacement = P * 1 / (((0.0713-0.0487)**2)*200e9 + ((A-((0.0713-0.0487)**2))*25e9))
theoretical_displacement_rebar = P * 1 / (((0.0713-0.0487)**2.0)*200e9)
measured_strain = strain_res[:, 0, 0]

lo = int(measured_strain.shape[0]*3/8)
hi = int(measured_strain.shape[0]*5/8)

end_node_mask = node_positions[:, 0] > 0.999
y = node_positions[:, 1]
z = node_positions[:, 2]
rebar_mask = (
    (y >= 0.0486) & (y <= 0.0713) &
    (z >= 0.0486) & (z <= 0.0713)
)

counts = np.unique_counts(E_est)
# print('counts',np.max(counts.counts), counts.values[np.argmax(counts.counts)])

actual_disp_end = np.mean(disp_meas[rebar_mask, 0][-4:])
integrated_strain = strain_res[rebar_mask,0,0].sum()*dx/4

print(f"Theoretical Displacement at end (concrete and rebar): {theoretical_displacement:.6e}")
print(f"Theoretical Displacement at end (rebar only): {theoretical_displacement_rebar:.6e}")
print(f"Simulated Displacement at end (u_L): {actual_disp_end:.6e}")
print(f"Simulated Integrated strain (sum(epsilon) * dx): {integrated_strain:.6e}")
print(f"Measured Integrated strain (sum(epsilon) * dx): {strain_meas.sum()*0.0013:.6e}")
# print(f"Difference: {abs(actual_disp_end - integrated_strain):.6e}")
# print(example.strain_est.numpy().shape)
# print(example.positions_sensors_np.shape)

x_s = example.positions_sensors_np[:, 0]
x_s_rounded = np.round(x_s, decimals=6)

x_s_unique = np.unique(x_s_rounded)
eps_est_avg = []

for xi in x_s_unique:
    mask_x = x_s_rounded == xi
    eps_est_avg.append(np.mean(np.array(strain_best)[mask_x]))
eps_est_avg = np.array(eps_est_avg)


end_node_mask = node_positions[:, 0] > 0.999
x = node_positions[:, 0]
y = node_positions[:, 1]
z = node_positions[:, 2]
rebar_mask = (
    (y >= 0.0486) & (y <= 0.0713) &
    (z >= 0.0486) & (z <= 0.0713)
)
x_rb = x[rebar_mask]
eps_xx_res  = strain_res[:, 0, 0]
eps_res_rb  = eps_xx_res[rebar_mask]
x_rounded = np.round(x_rb, decimals=6)

x_unique = np.unique(x_rounded)

eps_res_avg = []

for xi in x_unique:
    mask_x = x_rounded == xi
    eps_res_avg.append(np.mean(eps_res_rb[mask_x]))

eps_res_avg  = np.array(eps_res_avg)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(16, 16))

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
ax.set_title(f'Adam Learning Curve')#, lr={lr}')
result_dict = {
        "E_hist" : example.E_hist,
        "losses" : losses,
    }
exp_name = f"3d_prism_nd_limits_s{specimen}_ls{loadstep}_weights_{[str(i) for i in weights]}_freeze_{freeze_rebar}_slice_{opt_slice}_{data}_lr_{lr}_steps_{step}"
with open(f"results/{exp_name}.json", "w") as outfile: 
    json.dump(result_dict, outfile)

ax = axes[2]
ax.plot(strain_meas_x, strain_meas, label="Measured strain", lw=2)
ax.plot(x_s_unique, eps_est_avg,  label="Estimated strain", lw=2, linestyle="--")
ax.hlines(theoretical_displacement_rebar, 0, 1, color='r', label="Theoretical Max Rebar Strain")
ax.hlines(theoretical_displacement, 0, 1, color='0.5', label="Theoretical Min Combined Strain")
ax.set_xlabel("x")
ax.set_ylabel(r"$\varepsilon_{xx}$")
ax.set_title("Longitudinal strain in rebar (y–z averaged)")
ax.legend()

plt.savefig(f"figures/3d_prism_nd_limits/{exp_name}_LC.png", dpi=300)
# plt.show()





vmin = E_min
vmax = 27.0e9

import pyvista

cells, types = example._u_field_meas.space.vtk_cells()
nodes = example._u_field_meas.space.node_positions().numpy()
values = example._E_field_meas.dof_values.numpy()

# Visualize with pyvista
grid = pyvista.UnstructuredGrid(cells, types, nodes)

plotter = pyvista.Plotter(off_screen=True)
grid.cell_data["values"] = E_meas
plotter.add_mesh(grid, clim=[vmin, vmax], show_edges=True)
plotter.add_title("True Elastic Field")
plotter.show_axes()


grid.cell_data["values"] = E_best
centers = grid.cell_centers().points
x = centers[:,0]
y = centers[:,1]
z = centers[:,2]

z_unique = np.unique(z)


fig, axes = plt.subplots(len(z_unique)+3, 1, figsize=(12, 12), layout='constrained')

im_E_list = []
im_strain_list = []

# --- Shared scale for E fields ---
vmin_E = vmin
vmax_E = vmax

# --- Shared scale for strain fields ---
vmin_strain = min(np.nanpercentile(Exx_A, 1), np.nanpercentile(Exx_B, 1))
vmax_strain = max(np.nanpercentile(Exx_A, 99), np.nanpercentile(Exx_B, 99))

# --- 3D slices (E fields) ---
for i, zi in enumerate(z_unique):
    mask = z == zi
    E_val = grid.cell_data["values"][mask]
    coords = centers[mask][:, :2]

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)

    grid_E = np.full((len(y_unique), len(x_unique)), np.nan)

    x_idx = np.searchsorted(x_unique, x_vals)
    y_idx = np.searchsorted(y_unique, y_vals)

    grid_E[y_idx, x_idx] = E_val

    ax = axes[i]
    im = ax.imshow(
        grid_E,
        origin='lower',
        extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
        cmap='jet',
        vmin=vmin_E,
        vmax=vmax_E,
        aspect='auto'
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'z = {zi:.3f}')

    im_E_list.append(im)

# --- Strain images ---
idx_A = len(z_unique)
idx_B = len(z_unique) + 1

imA = axes[idx_A].imshow(
    Exx_A,
    origin='lower',
    cmap='jet',
    vmin=vmin_strain,
    vmax=vmax_strain
)
axes[idx_A].set_title(r'$\epsilon_{xx}$, Side A')

imB = axes[idx_B].imshow(
    Exx_B,
    origin='lower',
    cmap='jet',
    vmin=vmin_strain,
    vmax=vmax_strain
)
axes[idx_B].set_title(r'$\epsilon_{xx}$, Side B')

im_strain_list.extend([imA, imB])

# --- Colorbar for E fields ---
cbar_E = fig.colorbar(
    im_E_list[0],
    ax=axes[:len(z_unique)],
    orientation='horizontal',
    pad=0.02,
    aspect=50
)
cbar_E.set_label("Young's Modulus (Pa)")

# --- Colorbar for strain fields ---
cbar_strain = fig.colorbar(
    im_strain_list[0],
    ax=axes[len(z_unique):len(z_unique)+2],
    orientation='horizontal',
    pad=0.02,
    aspect=50
)
cbar_strain.set_label(r'$\epsilon_{xx}$')

# Get reference extent from slices
x_min = x_unique.min()
x_max = x_unique.max()
y_min = y_unique.min()
y_max = y_unique.max()

# --- Strain images ---
imA = axes[idx_A].imshow(
    Exx_A,
    origin='lower',
    cmap='jet',
    vmin=vmin_strain,
    vmax=vmax_strain,
    extent=[x_min, x_max, y_min, y_max],  # 🔥 key line
    aspect='auto'
)
axes[idx_A].set_title(r'$\epsilon_{xx}$, Side A')

imB = axes[idx_B].imshow(
    Exx_B,
    origin='lower',
    cmap='jet',
    vmin=vmin_strain,
    vmax=vmax_strain,
    extent=[x_min, x_max, y_min, y_max],  # 🔥 key line
    aspect='auto'
)
axes[idx_B].set_title(r'$\epsilon_{xx}$, Side B')

# --- Bottom plot ---
ax = axes[len(z_unique)+2]
ax.plot(strain_meas_x, strain_meas, label="Measured strain", lw=2)
ax.plot(x_s_unique, eps_est_avg, label="Estimated strain", lw=2, linestyle="--")
ax.hlines(theoretical_displacement_rebar, 0, 1, color='r', label="Theoretical Max Rebar Strain")
ax.hlines(theoretical_displacement, 0, 1, color='0.5', label="Theoretical Min Combined Strain")
ax.set_xlabel("x")
ax.set_ylabel(r"$\varepsilon_{xx}$")
ax.set_title("Longitudinal strain in rebar (y–z averaged)")
ax.legend()


    

plt.savefig(f"figures/3d_prism_nd_limits/{exp_name}_E.png", dpi=300)



# --- Node data ---
nodes = node_positions  # (N, 3)
x_nodes = nodes[:, 0]
y_nodes = nodes[:, 1]
z_nodes = nodes[:, 2]

# --- Simulated strain (nodal) ---
strain_xx = strain_res[:, 0, 0]

# Robust z slicing (handle float precision)
z_unique = np.unique(np.round(z_nodes, decimals=6))
n_slices = len(z_unique)

fig, axes = plt.subplots(n_slices + 3, 1, figsize=(12, 12), layout='constrained')

im_strain3D_list = []
im_dic_list = []

# --- Scales ---
vmin_sim = np.nanpercentile(strain_xx, 1)
vmax_sim = np.nanpercentile(strain_xx, 99)

vmin_dic = min(np.nanpercentile(Exx_A, 1), np.nanpercentile(Exx_B, 1))
vmax_dic = max(np.nanpercentile(Exx_A, 99), np.nanpercentile(Exx_B, 99))

# --- 3D strain slices (NODE-BASED) ---
for i, zi in enumerate(z_unique):
    mask = np.isclose(z_nodes, zi)

    strain_val = strain_xx[mask]
    x_vals = x_nodes[mask]
    y_vals = y_nodes[mask]

    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)

    grid_strain = np.full((len(y_unique), len(x_unique)), np.nan)

    x_idx = np.searchsorted(x_unique, x_vals)
    y_idx = np.searchsorted(y_unique, y_vals)

    grid_strain[y_idx, x_idx] = strain_val

    ax = axes[i]
    im = ax.imshow(
        grid_strain,
        origin='lower',
        extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
        cmap='jet',
        vmin=vmin_sim,
        vmax=vmax_sim,
        aspect='auto'
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Simulated $\\epsilon_{{xx}}$, z = {zi:.3f}')

    im_strain3D_list.append(im)

# --- Use last slice extent for alignment ---
x_min, x_max = x_unique.min(), x_unique.max()
y_min, y_max = y_unique.min(), y_unique.max()

# --- DIC strain images ---
idx_A = n_slices
idx_B = n_slices + 1

imA = axes[idx_A].imshow(
    Exx_A,
    origin='lower',
    cmap='jet',
    vmin=vmin_dic,
    vmax=vmax_dic,
    extent=[x_min, x_max, y_min, y_max],
    aspect='auto'
)
axes[idx_A].set_title(r'DIC $\epsilon_{xx}$, Side A')

imB = axes[idx_B].imshow(
    Exx_B,
    origin='lower',
    cmap='jet',
    vmin=vmin_dic,
    vmax=vmax_dic,
    extent=[x_min, x_max, y_min, y_max],
    aspect='auto'
)
axes[idx_B].set_title(r'DIC $\epsilon_{xx}$, Side B')

im_dic_list.extend([imA, imB])

# --- Colorbar: simulated strain ---
cbar_sim = fig.colorbar(
    im_strain3D_list[0],
    ax=axes[:n_slices],
    orientation='horizontal',
    pad=0.02,
    aspect=50
)
cbar_sim.set_label(r"Simulated $\epsilon_{xx}$")

# --- Colorbar: DIC strain ---
cbar_dic = fig.colorbar(
    im_dic_list[0],
    ax=axes[n_slices:n_slices+2],
    orientation='horizontal',
    pad=0.02,
    aspect=50
)
cbar_dic.set_label(r"DIC $\epsilon_{xx}$")

# --- Bottom plot ---
ax = axes[n_slices + 2]
ax.plot(strain_meas_x, strain_meas, label="Measured strain", lw=2)
ax.plot(x_s_unique, eps_est_avg, label="Estimated strain", lw=2, linestyle="--")
ax.hlines(theoretical_displacement_rebar, 0, 1, color='r', label="Theoretical Max Rebar Strain")
ax.hlines(theoretical_displacement, 0, 1, color='0.5', label="Theoretical Min Combined Strain")
ax.set_xlabel("x")
ax.set_ylabel(r"$\varepsilon_{xx}$")
ax.set_title("Longitudinal strain in rebar (y–z averaged)")
ax.legend()

plt.savefig(f"figures/3d_prism_nd_limits/{exp_name}_strain.png", dpi=300)

# --- SAVE DATA FOR PLOTTING ---
import os

# Create directory if it doesn't exist
save_dir = "plot_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Construct the filename (using the same exp_name used for figures)
save_path = os.path.join(save_dir, f"{exp_name}_data.npz")

np.savez(
    save_path,
    # Metadata/Arguments
    metadata={
        "specimen": specimen,
        "loadstep": loadstep,
        "freeze_rebar": freeze_rebar,
        "opt_slice": opt_slice,
        "lr": lr,
        "weights": weights,
        "load": load
    },
    # Optimization History
    losses=np.array(losses),
    params=np.array(params),
    
    # Experimental Data
    strain_meas=strain_meas,
    strain_meas_x=strain_meas_x,
    Exx_A=Exx_A,
    Exx_B=Exx_B,
    
    # Geometry and Fields
    node_positions=node_positions,
    E_best=E_best,
    strain_best=strain_best,
    strain_res=strain_res,
    
    # Processed plotting variables (averages/benchmarks)
    x_s_unique=x_s_unique,
    eps_est_avg=eps_est_avg,
    theoretical_displacement=theoretical_displacement,
    theoretical_displacement_rebar=theoretical_displacement_rebar,
    
    # Mesh data for PyVista
    vtk_cells=cells,
    vtk_types=types
)

print(f"\nPlotting data saved successfully to: {save_path}")
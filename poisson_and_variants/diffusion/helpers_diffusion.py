import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from poisson.helpers import *

def z_cov(x0: float, x1: float, y0: float, y1: float) -> float:
    """
    Args:
        x0, x1: Coordinates of the first point
        y0, y1: Coordinates of the second point
    Returns:
        Exponential Covariance between the two points
    """
    SIGMA = 5.0
    l = 0.1
    # Exponential Covariance Function
    return SIGMA**2 * np.exp(-np.sqrt((x0 - y0)**2 + (x1 - y1)**2) / l)

class RandomFieldZ():
    """Class to represent the random field Z with eigenvalues and eigenvectors."""
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array, basis_functions: list[ConstBasisFunction], N: int, J: int):
        """
        Args:
            eigenvalues: Eigenvalues of the covariance matrix
            eigenvectors: Eigenvectors of the covariance matrix
            basis_functions: List of basis functions
            N: Number of basis functions
            J: KL truncation level
        """
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J

    def __call__(self, x: np.array, xi: np.array) -> float:
        """Evaluate the random field Z at point x with random variables xi."""
        index_supported_basis_function = self.find_supported_basis_function(x)
        if index_supported_basis_function is None:
            return 0
        return sum([np.sqrt(self.eigenvalues[j]) * self.eigenvectors[index_supported_basis_function, j] * xi[j] for j in range(len(xi))])
    
    def find_supported_basis_function(self, x: np.array) -> int:
        """Find the index of the only basis function that supports the point x."""
        for i, basis_function in enumerate(self.basis_functions):
            if basis_function.function(x) == 1:
                return i
        return None

def z_calculate_random_field_eigenpairs(mesh_resolution: int, z_cov: callable) -> RandomFieldZ:
    """Calculate the eigenpairs of the covariance matrix for the random field Z.

    Args:
        mesh_resolution: Resolution of the mesh
        z_cov: Covariance function for the random field Z
    Returns:
        RandomFieldZ object with eigenvalues and eigenvectors
    """
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "DG", 0)
    N = V.dim()

    basis_functions = []
    for i in range(N):
        basis_function = fe.Function(V)
        basis_function.vector()[i] = 1.0
        basis_function.set_allow_extrapolation(True)

        cell_index = V.dofmap().cell_dofs(i)[0]
        cell = fe.Cell(mesh, cell_index)
        vertex_coords = np.array([np.array(mesh.coordinates()[vertex]) for vertex in cell.entities(0)])
        basis_functions.append(ConstBasisFunction(basis_function, vertex_coords))
    
    C = np.zeros((N, N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if j <= i:
                # Here we use that each block is symmetric because of the symmetry of the covariance functions
                C[i, j] = C[j, i] = get_C_entry(z_cov, basis_function_i, basis_function_j)
    
    M = np.zeros((N, N))
    for i, basis_function in enumerate(basis_functions):
        integrand = basis_function_i.function * basis_function_i.function * fe.dx
        M[i, i] = fe.assemble(integrand)
        
    J = N # Number of eigenvectors
    eigenvalues, eigenvectors = eig(C, M)
    eigenvalues = eigenvalues[:J].real
    eigenvectors = eigenvectors[:, :J].real
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Eliminate negative eigenvalues
    for index, sorted_eigenvalue in enumerate(sorted_eigenvalues):
        if sorted_eigenvalue < 0:
            sorted_eigenvalues[index] = 0

    return RandomFieldZ(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J)

def a_hat_random_field(x: np.array, randomFieldV: RandomFieldV, randomFieldZ: RandomFieldZ, xi_v: np.array, xi_z: np.array) -> float:
    """Calculate the random field a_hat at point x.

    Args:
        x: Coordinates of the point
        randomFieldV: Random field V object
        randomFieldZ: Random field Z object
        xi_v: Random variables for the random field V
        xi_z: Random variables for the random field Z
        Returns:
        a_hat: Random field a_hat = exp(z(V(x))) with mean of z = 0
    """
    x_hat = randomFieldV(x, xi_v)
    z = randomFieldZ(x_hat, xi_z)
    a = np.exp(z)
    return a

class BExpression(fe.UserExpression):
    """Class to represent the B matrix in the diffusion equation."""
    def __init__(self, randomFieldV: RandomFieldV, jacobianV_fixed_xi: JacobianVFixedXi, randomFieldZ: RandomFieldZ, xi_v: np.array, xi_z: np.array, **kwargs):
        """
        Args:
            randomFieldV: Random field V object
            jacobianV_fixed_xi: Jacobian of the random field V
            randomFieldZ: Random field Z object
            xi_v: Random variables for the random field V
            xi_z: Random variables for the random field Z
        """
        super().__init__(**kwargs)
        self.randomFieldV = randomFieldV
        self.jacobianV_fixed_xi = jacobianV_fixed_xi
        self.randomFieldZ = randomFieldZ
        self.xi_v = xi_v
        self.xi_z = xi_z

    def eval(self, values, x: np.array) -> None:
        """Evaluate the B matrix at point x."""
        J_x = self.jacobianV_fixed_xi(x)
        inv_JTJ = np.linalg.inv(J_x.T @ J_x)
        det_J = np.linalg.det(J_x)
        a_hat = a_hat_random_field(x, self.randomFieldV, self.randomFieldZ, self.xi_v, self.xi_z)
        B_x = a_hat * inv_JTJ * det_J
        values[0] = B_x[0, 0]
        values[1] = B_x[0, 1]
        values[2] = B_x[1, 0]
        values[3] = B_x[1, 1]

    def value_shape(self):
        return (2, 2)
    
def solve_diffusion_poisson_for_given_sample(mesh_resolution: int, f: float, randomFieldV: RandomFieldV, jacobianV: JacobianV, randomFieldZ: RandomFieldZ, xi_v: np.array, xi_z: np.array) -> fe.Function:
    """Solve the diffusion equation for a given sample.

    Args:
        mesh_resolution: Resolution of the mesh
        f: Right-hand side of the diffusion equation
        randomFieldV: Random field V object
        jacobianV: Jacobian of the random field V
        randomFieldZ: Random field Z object
        xi_v: Random variables for the random field V
        xi_z: Random variables for the random field Z
    Returns:
        u_sol: Solution of the diffusion equation
    """
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 1)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    jacobianV_fixed_xi = JacobianVFixedXi(jacobianV, xi_v)
    B_expr = BExpression(randomFieldV, jacobianV_fixed_xi, randomFieldZ, xi_v, xi_z, degree=2)
    a = fe.inner(fe.dot(B_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV_fixed_xi, degree=2)
    L = f * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    u_sol.set_allow_extrapolation(True)
    return u_sol

# Monte Carlo Simulation

def diffusion_save_results_to_csv(u_sols_evaluated: np.array, fem_res: int, kl_res: int) -> None:
    """Save the results of the diffusion equation to a CSV file.

    Args:
        u_sols_evaluated: Evaluated solutions of the diffusion equation
        fem_res: Finite element mesh resolution
        kl_res: Karhunen-Loeve mesh resolution
    """
    i = 0
    samples_filename = f'mc_data_storage/samples_femres_{fem_res}_klres_{kl_res}_{i}.csv'


    # Increment i until a non-existing filename is found
    while os.path.exists(samples_filename):
        i += 1
        samples_filename = f'mc_data_storage/samples_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    
    # Save u_sols_evaluated to the found filename
    np.savetxt(samples_filename, u_sols_evaluated, delimiter=',', fmt="%.18e")
    print(f"u_sols_evaluated saved to {samples_filename}")

def diffusion_calculate_samples_and_save_results(mc_samples: int, fem_res: int, kl_res: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None, randomFieldZ: RandomFieldZ = None) -> None:
    """Calculate samples and save results to files.

    Args:
        mc_samples: Number of Monte Carlo samples
        fem_res: Finite element mesh resolution
        kl_res: Karhunen-Loeve mesh resolution
        randomFieldV: Random field V object
        jacobianV: Jacobian of the random field V
        randomFieldZ: Random field Z object
    """
    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    if not randomFieldZ:
        randomFieldZ = z_calculate_random_field_eigenpairs(kl_res, z_cov)

    # Take fem_res for saving scheme
    mesh = mshr.generate_mesh(DOMAIN, fem_res)
    V = fe.FunctionSpace(mesh, "CG", 1)
    dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

    u_sols = []
    u_sols_evaluated = np.zeros((mc_samples, len(dof_coordinates)))
    for i in range(mc_samples):
        print(f"Iteration {i+1}/{mc_samples}")
        xi_v = np.random.uniform(-np.sqrt(3), np.sqrt(3), randomFieldV.J)
        xi_z = np.random.normal(0, 1, randomFieldZ.J)
        u_sols.append(solve_diffusion_poisson_for_given_sample(fem_res, RHS_F, randomFieldV, jacobianV, randomFieldZ, xi_v, xi_z))
        for j, point_coords in enumerate(dof_coordinates):
            u_sols_evaluated[i, j] = u_sols[i](fe.Point(point_coords))
    diffusion_save_results_to_csv(u_sols_evaluated, fem_res=fem_res, kl_res=kl_res)

def diffusion_analyse_two_resolutions_from_data_u_hat(resolution_sparse: int, resolution_fine: int, P_hat: fe.Point) -> None:
    """Analyse the results of two different resolutions and plot the results.

    Args:
        resolution_sparse: FEM resolution of both meshes and KL resolution of the sparse mesh
        resolution_fine: KL resolution of the fine mesh
        P_hat: Point at which the results are evaluated
    """
    # Here the sparse resolution is used for both fem_res and the fine resolution is used just for the kl_res of the fine solution
    # So for both functionspaces the sparse mesh resolution is used for the mesh
    data_sparse = np.genfromtxt(f'mc_data_storage/samples_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    data_fine = np.genfromtxt(f'mc_data_storage/samples_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')

    mesh_sparse = mshr.generate_mesh(DOMAIN, resolution_sparse)
    V_sparse = fe.FunctionSpace(mesh_sparse, "CG", 1)

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [32]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= data_sparse.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------

    print(f"Sample size fine solution: {data_fine.shape[0]}")
    print(f"Samples sizes sparse solution: {mc_sample_sizes}")

    mean_sol_fine = fe.Function(V_sparse)
    mean_sol_fine.set_allow_extrapolation(True)
    mean_sol_fine.vector()[:] = np.mean(data_fine, axis=0)

    var_sol_fine = fe.Function(V_sparse)
    var_sol_fine.set_allow_extrapolation(True)
    var_sol_fine.vector()[:] = np.var(data_fine, axis=0)

    lower_confidence_bound_fine = mean_sol_fine(P_hat) - 1.96 * np.sqrt(var_sol_fine(P_hat) / data_fine.shape[0])
    upper_confidence_bound_fine = mean_sol_fine(P_hat) + 1.96 * np.sqrt(var_sol_fine(P_hat) / data_fine.shape[0])

    u_sols_sparse = []
    u_sols_sparse_P_hat_means = []
    u_sols_sparse_P_hat_vars = []
    lower_confidence_bounds = []
    upper_confidence_bounds = []
    L2_errors = []
    H1_errors = []
    for mc_sample_index, mc_sample_size in enumerate(mc_sample_sizes):
        mean_sol_sparse = fe.Function(V_sparse)
        mean_sol_sparse.set_allow_extrapolation(True)

        # Calculate mean solution
        if mc_sample_index == 0:
            data_sparse_sample = data_sparse[:mc_sample_size]
            mean_sol_sparse.vector()[:] = np.mean(data_sparse_sample, axis=0) # actually the same
        else:
            data_sparse_sample = data_sparse[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]
            mean_sol_sparse.vector()[:] = np.mean(data_sparse_sample, axis=0)
        u_sols_sparse.append(mean_sol_sparse)
        u_sols_sparse_P_hat_means.append(mean_sol_sparse(P_hat))

        # Calculate variance in point P_hat
        var_sol_sparse = fe.Function(V_sparse)
        var_sol_sparse.set_allow_extrapolation(True)
        var_sol_sparse.vector()[:] = np.var(data_sparse_sample, axis=0)
        u_sols_sparse_P_hat_vars.append(var_sol_sparse(P_hat))

        # Calculate confidence intervals
        lower_confidence_bounds.append(u_sols_sparse_P_hat_means[-1] - 1.96 * np.sqrt(u_sols_sparse_P_hat_vars[-1] / mc_sample_size))
        upper_confidence_bounds.append(u_sols_sparse_P_hat_means[-1] + 1.96 * np.sqrt(u_sols_sparse_P_hat_vars[-1] / mc_sample_size))
        
        # Calculate L2 and H1 errors
        L2_errors.append(fe.errornorm(mean_sol_fine, mean_sol_sparse, 'L2'))
        H1_errors.append(fe.errornorm(mean_sol_fine, mean_sol_sparse, 'H1'))

    # Plot fine solution
    x_coords = mesh_sparse.coordinates()[:, 0]
    y_coords = mesh_sparse.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]
    z_values_fine_mean = []
    z_values_fine_var = []

    for i in range(len(x_coords)):
        z_values_fine_mean.append(mean_sol_fine(x_coords[i], y_coords[i]))
        z_values_fine_var.append(var_sol_fine(x_coords[i], y_coords[i]))

    grid_z_mean = griddata((x_coords, y_coords), z_values_fine_mean, (grid_x, grid_y), method='linear')
    fig_mean, ax = plt.subplots(figsize=(10, 8))
    cp = ax.contourf(grid_x, grid_y, grid_z_mean, levels=100, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(r'Mean estimation $\hat{u}(\hat{x},\omega)$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.scatter(P_hat.x(), P_hat.y(), color='red', s=100, label=r'$\hat{P}$')
    ax.legend(loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.show()

    grid_z_var = griddata((x_coords, y_coords), z_values_fine_var, (grid_x, grid_y), method='linear')
    fig_var, ax = plt.subplots(figsize=(10, 8))
    cp = ax.contourf(grid_x, grid_y, grid_z_var, levels=100, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    # cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    cbar.ax.yaxis.get_offset_text().set_fontsize(20)
    ax.set_title(r'Variance estimation $\hat{u}(\hat{x},\omega)$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.scatter(P_hat.x(), P_hat.y(), color='red', s=100, label=r'$\hat{P}$')
    ax.legend(loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.show()


    # Plot means
    fig_var, ax = plt.subplots(figsize=(10, 8))
    ax.plot(mc_sample_sizes, u_sols_sparse_P_hat_means, 'bo', marker='x', linestyle='None', label='Means', markersize=10)
    ax.fill_between(mc_sample_sizes, lower_confidence_bounds, upper_confidence_bounds, alpha=0.2, label='95% Confidence Interval')
    ax.axhline(y=mean_sol_fine(P_hat), color='r', linestyle='-', label='Mean reference solution')
    ax.fill_between(mc_sample_sizes, lower_confidence_bound_fine, upper_confidence_bound_fine, alpha=0.2, color='red', label='95% Confidence Interval reference solution')
    ax.set_xscale('log')
    ax.set_xlabel('MC Samples', fontsize=24)
    ax.set_ylabel('Mean', fontsize=24)
    ax.legend(loc='upper left', fontsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot variance
    fig_var, ax = plt.subplots(figsize=(10, 8))
    ax.plot(mc_sample_sizes, u_sols_sparse_P_hat_vars, 'go', marker='x', linestyle='None', label='Variance', markersize=10)
    ax.axhline(y=var_sol_fine(P_hat), color='r', linestyle='-', label='Variance reference solution')
    ax.set_xscale('log')
    ax.set_xlabel('MC Samples', fontsize=24)
    ax.set_ylabel('Variance', fontsize=24)
    ax.legend(loc='upper left', fontsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.get_offset_text().set_fontsize(20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plots L2 and H1 errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

    ax1.plot(mc_sample_sizes, L2_errors, 'bo', marker='x', label=r'$L^2$ Error')
    ax1.set_xscale('log')
    #ax1.set_yscale('log')
    ax1.set_xlabel('MC Samples')
    ax1.set_ylabel(r'$L^2$ Error')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(mc_sample_sizes, H1_errors, 'bo', marker='x', label=r'$H^1$ Error')
    ax2.set_xscale('log')
    #ax2.set_yscale('log')
    ax2.set_xlabel('MC Samples')
    ax2.set_ylabel(r'$H^1$ Error')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.suptitle(r'$L^2$ and $H^1$ Errors of $\hat{u}(\hat{x}, \omega)$ to the reference solution')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Sobol Indices

def diffusion_sobol_run_samples_and_save(mc_sample_size: int, fem_res: int, kl_res: int, size_xi_v: int, size_xi_z: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None, randomFieldZ: RandomFieldZ = None) -> None:
    """Calculate Sobol indices and save results to files.

    Args:
        mc_sample_size: Number of Monte Carlo samples
        fem_res: Finite element mesh resolution
        kl_res: Karhunen-Loeve mesh resolution
        size_xi_v: Size of the xi_v vector
        size_xi_z: Size of the xi_z vector
        randomFieldV: Random field V object
        jacobianV: Jacobian of the random field V
        randomFieldZ: Random field Z object     
    """
    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    if not randomFieldZ:
        randomFieldZ = z_calculate_random_field_eigenpairs(kl_res, z_cov)
    
    # Collect points P
    N = len(randomFieldV.basis_functions)
    
    # Sample all xis, not only the ones that are investigated
    len_xi_total = randomFieldV.J + randomFieldZ.J
    A = np.zeros((mc_sample_size, len_xi_total))
    B = np.zeros((mc_sample_size, len_xi_total))

    for xi_v_index in range(randomFieldV.J):
        A[:, xi_v_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_v_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for xi_z_index in range(randomFieldZ.J):
        A[:, randomFieldV.J + xi_z_index] = np.random.normal(0, 1, size=mc_sample_size)
        B[:, randomFieldV.J + xi_z_index] = np.random.normal(0, 1, size=mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        print(f"First loop Iteration {m+1}/{mc_sample_size}")
        u_sol_A = solve_diffusion_poisson_for_given_sample(fem_res, RHS_F, randomFieldV, jacobianV, randomFieldZ, A[m, :randomFieldV.J], A[m, randomFieldV.J:])
        u_sol_B = solve_diffusion_poisson_for_given_sample(fem_res, RHS_F, randomFieldV, jacobianV, randomFieldZ, B[m, :randomFieldV.J], B[m, randomFieldV.J:])

        for basis_function_index, basis_function in enumerate(randomFieldV.basis_functions):
            transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
            active_quad_points = QUAD_POINTS_2DD_6
            quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
            int_A = 0
            int_B = 0
            for quad_point in quad_points:
                int_A += u_sol_A(quad_point.point) * quad_point.weight
                int_B += u_sol_B(quad_point.point) * quad_point.weight
            f_A[m, basis_function_index] = int_A
            f_B[m, basis_function_index] = int_B
        
    f_C_is = []

    for i in range(size_xi_v + size_xi_z):
        C_i = np.zeros((mc_sample_size, len_xi_total))
        for xi_v_index in range(randomFieldV.J):
            if xi_v_index == i and xi_v_index < size_xi_v:
                C_i[:, xi_v_index] = A[:, xi_v_index]
            else:
                C_i[:, xi_v_index] = B[:, xi_v_index]

        for xi_z_index in range(randomFieldZ.J):
            if size_xi_v + xi_z_index == i and xi_z_index < size_xi_z:
                C_i[:, randomFieldV.J + xi_z_index] = A[:, randomFieldV.J + xi_z_index]
            else:
                C_i[:, randomFieldV.J + xi_z_index] = B[:, randomFieldV.J + xi_z_index]
        
        f_C_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            print(f"Second loop xi {i+1} of {size_xi_v + size_xi_z}, Iteration {m+1}/{mc_sample_size}")
            u_sol_C_i = solve_diffusion_poisson_for_given_sample(fem_res, RHS_F, randomFieldV, jacobianV, randomFieldZ, C_i[m, :randomFieldV.J], C_i[m, randomFieldV.J:])
            for basis_function_index, basis_function in enumerate(randomFieldV.basis_functions):
                transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
                active_quad_points = QUAD_POINTS_2DD_6
                quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
                int_C_i = 0
                for quad_point in quad_points:
                    int_C_i += u_sol_C_i(quad_point.point) * quad_point.weight
                f_C_i[m, basis_function_index] = int_C_i
        f_C_is.append(f_C_i)
        
    base_path = f'sobol_data_storage/femres_{fem_res}_klres_{kl_res}_size_xi_v_{size_xi_v}_size_xi_z_{size_xi_z}'
    f_A_path = os.path.join(base_path, 'f_A.npy')
    f_B_path = os.path.join(base_path, 'f_B.npy')

    if os.path.exists(f_A_path):
        f_A_existing = np.load(f_A_path)
        f_A = np.concatenate((f_A_existing, f_A), axis=0)
    if os.path.exists(f_B_path):
        f_B_existing = np.load(f_B_path)
        f_B = np.concatenate((f_B_existing, f_B), axis=0)

    np.save(f_A_path, f_A)
    np.save(f_B_path, f_B)

    for i in range(size_xi_v + size_xi_z):
        f_C_i_path = os.path.join(base_path, f'f_C_{i}.npy')
        if os.path.exists(f_C_i_path):
            f_C_i_existing = np.load(f_C_i_path)
            f_C_is[i] = np.concatenate((f_C_i_existing, f_C_is[i]), axis=0)
        np.save(f_C_i_path, f_C_is[i])

    print("Sobol data saved")

def diffusion_sobol_calc_indices_from_data(fem_res: int, kl_res: int, size_xi_v: int, size_xi_z: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None) -> tuple[np.array, np.array, int]:
    """Calculate Sobol indices from data.

    Args:
        fem_res: Finite element mesh resolution
        kl_res: Karhunen-Loeve mesh resolution
        size_xi_v: Size of the xi_v vector
        size_xi_z: Size of the xi_z vector
        randomFieldV: Random field V object
        jacobianV: Jacobian of the random field V
    """
    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    base_path = f'sobol_data_storage/femres_{fem_res}_klres_{kl_res}_size_xi_v_{size_xi_v}_size_xi_z_{size_xi_z}'
    f_A_path = os.path.join(base_path, 'f_A.npy')
    f_B_path = os.path.join(base_path, 'f_B.npy')

    f_A = np.load(f_A_path)
    f_B = np.load(f_B_path)

    S_single = np.zeros(size_xi_v + size_xi_z)
    S_total = np.zeros(size_xi_v + size_xi_z)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions]

    for i in range(size_xi_v + size_xi_z):
        f_C_i_path = os.path.join(base_path, f'f_C_{i}.npy')
        f_C_i = np.load(f_C_i_path)

        var_g_q_i = (1 / f_A.shape[0]) * np.sum(f_A * f_C_i, axis=0) - f_0_squared
        var_g_q_i_tilde = (1 / f_A.shape[0]) * np.sum(f_B * f_C_i, axis=0) - f_0_squared
        var_g = (1 / f_A.shape[0]) * np.sum(f_A**2, axis=0) - f_0_squared

        numerator_s_single = np.sum(weights * var_g_q_i)
        numerator_s_total = np.sum(weights * var_g_q_i_tilde)
        denominator = np.sum(weights * var_g)

        S_single[i] = numerator_s_single / denominator
        S_total[i] = 1 - numerator_s_total / denominator

    return S_single, S_total, f_A.shape[0]

def diffusion_plot_sobols(S_single: np.array, S_total: np.array, mc_sample_size: int, size_xi_v: int) -> None:
    """Plot Sobol indices.

    Args:
        S_single: First order Sobol indices
        S_total: Total Sobol indices
        mc_sample_size: Number of Monte Carlo samples
        size_xi_v: Size of the xi_v vector
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    # Set width for each bar
    bar_width = 0.35
    ax.bar(np.arange(len(S_single)), S_single, width=bar_width, label='First Order')
    ax.bar(np.arange(len(S_single)) + bar_width, S_total, width=bar_width, label='Total Effect')
    x_labels = [fr"$\xi_{i+1}^V$" for i in range(size_xi_v)] + [fr"$\xi_{i+1}^Z$" for i in range(len(S_single) - size_xi_v)]
    ax.set_xticklabels(x_labels, fontsize=24)
    ax.set_xticks(np.arange(len(S_single)) + bar_width / 2)
    ax.set_ylabel('Sensitivity [-]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    ax.legend(fontsize=20)
    plt.show()
    print(f"Sample size: {mc_sample_size}")

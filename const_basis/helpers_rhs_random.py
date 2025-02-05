from helpers import *

# Test function for algorithms
def test_f(x):
    return x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] + 5 * x[4]

# Helpers Sensitivity Analysis

class RandomRHS(fe.UserExpression):
    def __init__(self, F1, F2, **kwargs):
        super().__init__(**kwargs)
        self.F1 = F1
        self.F2 = F2
    
    def eval(self, values, x):
        if np.linalg.norm(x) < 1:
            if x[0] <= 0:
                values[0] = self.F1
            else:
                values[0] = self.F2
        else:
            values[0] = 0
    
    def value_shape(self):
        return ()

class RandomRHSSecondAlternative(fe.UserExpression):
    # of the form F_1 \cdot \sin(\lVert x \rVert_2) + F_2 \cdot \cos(\lVert x \rVert_2)
    def __init__(self, F1, F2, **kwargs):
        super().__init__(**kwargs)
        self.F1 = F1
        self.F2 = F2
    
    def eval(self, values, x):
        values[0] = self.F1 * np.sin(np.sqrt(x[0]**2 + x[1]**2)) + self.F2 * np.cos(np.sqrt(x[0]**2 + x[1]**2))
    
    def value_shape(self):
        return ()

def solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, xi, F_samples):
    f_expr = RandomRHS(F_samples[0], F_samples[1], degree=2)
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 1)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    jacobianV_fixed_xi = JacobianVFixedXi(jacobianV, xi)
    A_expr = AExpression(jacobianV_fixed_xi, degree=2)
    a = fe.inner(fe.dot(A_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV_fixed_xi, degree=2)
    L = f_expr * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    u_sol.set_allow_extrapolation(True)
    return u_sol

def double_loop_mc(mc_sample_size, mesh_resolution_fem, P, indices, randomFieldV, jacobianV, size_of_xi):
    # Translate math indices to CS indices
    indices = [index - 1 for index in indices]

    # Implementation of the double loop MC estimation of specific Sobol Index
    mean_sols_point_P = np.zeros(mc_sample_size)
    for i in range(mc_sample_size):
        print(f"First loop: Outer MC loop iteration {i+1} / {mc_sample_size}")
        sols_point_P = []
        xi = np.zeros(size_of_xi)
        F_samples = np.zeros(2)
        for index in indices:
            if index < size_of_xi:
                xi[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
            else:
                F_samples[index - size_of_xi] = np.random.random(size=1)
        missing_indices = [i for i in range(size_of_xi + 1) if i not in indices]
        for k in range(mc_sample_size):
            for index in missing_indices:
                if index < size_of_xi:
                    xi[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                else:
                    F_samples[index - size_of_xi] = np.random.random(size=1)
            P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
            sols_point_P.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
        mean_sols_point_P[i] = np.mean(sols_point_P)
    var_A = np.var(mean_sols_point_P, ddof=1)
    sols_point_P = []
    for i in range(mc_sample_size):
        print(f"Second loop: {i+1} / {mc_sample_size}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        F_samples = np.random.random(size=2)
        P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
        sols_point_P.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
        
    var = np.var(sols_point_P, ddof=1)
    return var_A / var

def pick_freeze(mc_sample_size, mesh_resolution_fem, P, indices, randomFieldV, jacobianV, size_of_xi):
    # Translate math indices to CS indices
    indices = [index - 1 for index in indices]

    y = []
    y_tilde = []
    for i in range(mc_sample_size):
        print(f"First loop: {i+1} / {mc_sample_size}")
        xi = np.zeros(size_of_xi)
        xi_prime = np.zeros(size_of_xi)
        F_samples = np.zeros(2)
        F_samples_prime = np.zeros(2)
        for index in indices:
            if index < size_of_xi:
                sample = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                xi[index] = sample
                xi_prime[index] = sample
            else:
                sample = np.random.random(size=1)
                F_samples[index - size_of_xi] = sample
                F_samples_prime[index - size_of_xi] = sample
        for index in [i for i in range(size_of_xi + 1) if i not in indices]:
            if index < size_of_xi:
                xi[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                xi_prime[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
            else:
                F_samples[index - size_of_xi] = np.random.random(size=1)
                F_samples_prime[index - size_of_xi] = np.random.random(size=1)
        P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
        y.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
        P_prime_hat = inverse_mapping(P, randomFieldV, xi_prime, mesh_resolution_fem)
        y_tilde.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi_prime, F_samples_prime)(P_prime_hat))
    y_bar = np.mean(y)
    y_tilde_bar = np.mean(y_tilde)
    if mc_sample_size > 1:
        var_A = 1/(mc_sample_size - 1) * sum([(y[i] - y_bar) * (y_tilde[i] - y_tilde_bar) for i in range(mc_sample_size)])
    else:
        var_A = 0

    sols_point_P = []
    for i in range(mc_sample_size):
        print(f"Second loop: {i+1} / {mc_sample_size}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        F_samples = np.random.random(size=2)
        P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
        sols_point_P.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
    var = np.var(sols_point_P, ddof=1)
    return var_A / var

def rank(i, sorted_samples, samples):
    return sorted_samples.index(samples[i])
def rank_inv(i, sorted_samples, samples):
    return samples.index(sorted_samples[i])
def P_j(i, sorted_samples, samples):
    if rank(i, sorted_samples, samples) + 2 <= len(samples):
        return rank_inv(rank(i, sorted_samples, samples) + 1, sorted_samples, samples)
    else:
        return rank_inv(0, sorted_samples, samples)

def rank_statistics_permute(samples, y):
    sorted_samples = sorted(samples)
    helper = np.zeros((len(y), 3))
    for i in range(len(y)):
        helper[i, 0] = y[i]
        helper[i, 1] = P_j(i, sorted_samples, samples)
        helper[i, 2] = samples[P_j(i, sorted_samples, samples)]
    sorted_helper = helper[helper[:, 2].argsort()]
    return sorted_helper[:, 2], sorted_helper[:, 0]
    
def rank_statistics(mc_sample_size, mesh_resolution_fem, P, index, randomFieldV, jacobianV, size_of_xi):

    # Translate math index to CS index
    index -= 1

    y = []
    xis = np.zeros((mc_sample_size, size_of_xi))
    F_samples = np.zeros((mc_sample_size, 2))
    for i in range(mc_sample_size):
        print(f"Loop: {i+1} / {mc_sample_size}")
        xis[i, :] = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        F_samples[i, :] = np.random.random(size=2)
        P_hat = inverse_mapping(P, randomFieldV, xis[i, :], mesh_resolution_fem)
        y.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xis[i, :], F_samples[i, :])(P_hat))
    y_bar = np.mean(y)
    if index < size_of_xi:
        xi_after_permute, y_after_permute = rank_statistics_permute(xis[:, index].tolist(), y)
        V_hat_j = 1/mc_sample_size * sum([y_after_permute[i] * y_after_permute[P_j(i, sorted(xi_after_permute.tolist()), xi_after_permute.tolist())] for i in range(mc_sample_size)]) - y_bar**2
    else:
        F_after_permute, y_after_permute = rank_statistics_permute(F_samples[:, index - size_of_xi].tolist(), y)
        V_hat_j = 1/mc_sample_size * sum([y_after_permute[i] * y_after_permute[P_j(i, sorted(F_after_permute.tolist()), F_after_permute.tolist())] for i in range(mc_sample_size)]) - y_bar**2
    V_hat = 1/mc_sample_size * sum([y[i]**2 for i in range(mc_sample_size)]) - y_bar**2
    return V_hat_j / V_hat

def rhs_random_functional_valued_output_sobol_estimation_u_hat(mc_sample_size: int, mesh_resolution: int, size_of_xi: int) -> tuple[np.array, np.array]:

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    # Collect points P
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions]
    N = len(Ps)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions]
    
    # Sample all xis, not only the ones that are investigated
    len_xi_total = randomFieldV.J
    A = np.zeros((mc_sample_size, len_xi_total + 2))
    B = np.zeros((mc_sample_size, len_xi_total + 2))

    for xi_index in range(len_xi_total):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for F_index in range(2):
        A[:, len_xi_total + F_index] = np.random.random(size=mc_sample_size)
        B[:, len_xi_total + F_index] = np.random.random(size=mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_A = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, A[m, :len_xi_total], A[m, len_xi_total:])
        u_sol_B = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, B[m, :len_xi_total], B[m, len_xi_total:])
        for P_index, P in enumerate(Ps):
            f_A[m, P_index] = u_sol_A(P)
            f_B[m, P_index] = u_sol_B(P)
        
    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    S_single = np.zeros(size_of_xi + 2)
    S_total = np.zeros(size_of_xi + 2)

    for i in range(size_of_xi + 2):
        A_B_i = np.zeros((mc_sample_size, len_xi_total + 2))
        if i >= size_of_xi:
            # F cases
            rel_index = i - size_of_xi
            for param_index in range(len_xi_total + 2):
                if param_index == len_xi_total + rel_index:
                    A_B_i[:, param_index] = B[:, param_index]
                else:
                    A_B_i[:, param_index] = A[:, param_index]
        else:
            for param_index in range(len_xi_total + 2):
                if param_index == i:
                    A_B_i[:, param_index] = B[:, param_index]
                else:
                    A_B_i[:, param_index] = A[:, param_index]
        f_A_B_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            u_sol_A_B_i = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, A_B_i[m, :len_xi_total], A_B_i[m, len_xi_total:])
            for P_index, P in enumerate(Ps):
                f_A_B_i[m, P_index] = u_sol_A_B_i(P)
        
        var_g_i = (1 / mc_sample_size) * np.sum(f_B * (f_A_B_i - f_A), axis=0)
        E_var_g_i = (1 / mc_sample_size) * np.sum(f_A * (f_A - f_A_B_i), axis=0)
        var_g = (1 / mc_sample_size) * np.sum(f_A**2, axis=0) - f_0_squared

        numerator_s_single = np.sum(weights * var_g_i)
        numerator_s_total = np.sum(weights * E_var_g_i)
        denominator = np.sum(weights * var_g)

        S_single[i] = numerator_s_single / denominator
        S_total[i] = numerator_s_total / denominator

    return S_single, S_total

    fig, ax = plt.subplots(figsize=(10, 5))

    # Set width for each bar
    bar_width = 0.35

    ax.bar(np.arange(len(S_single)), S_single, width=bar_width, label='First Order')
    ax.bar(np.arange(len(S_single)) + bar_width, S_total, width=bar_width, label='Total Effect')
    x_labels = [fr"$\xi_{{{i+1}}}$" for i in range(len(S_single) - 2)] + [r"$F_1$", r"$F_2$"]
    ax.set_xticklabels(x_labels)
    ax.set_xticks(np.arange(len(S_single)) + bar_width / 2)
    ax.set_ylabel('Sensitivity [-]')
    ax.set_title(f'{title} Sample Size: {mc_sample_size}')
    ax.grid(True)
    ax.legend()
    plt.show()

def poisson_rhs_random_save_results_to_csv(u_sols_evaluated: np.array, xis: np.array, fem_res: int, kl_res: int):
    i = 0
    samples_filename = f'poisson_rhs_random_sample_storage/samples_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    # xis_filename = f'poisson_rhs_random_sample_storage/xis_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    
    # Increment i until a non-existing filename is found
    while os.path.exists(samples_filename):
        i += 1
        samples_filename = f'poisson_rhs_random_sample_storage/samples_femres_{fem_res}_klres_{kl_res}_{i}.csv'
        # xis_filename = f'poisson_rhs_random_sample_storage/xis_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    
    # Save u_sols_evaluated to the found filename
    np.savetxt(samples_filename, u_sols_evaluated, delimiter=',', fmt="%.18e")
    # np.savetxt(xis_filename, xis, delimiter=',', fmt="%.18e")
    print(f"u_sols_evaluated saved to {samples_filename}")

def poisson_rhs_random_calculate_samples_and_save_results(mc_samples: int, fem_res: int, kl_res: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None):

    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    # Take fem_res for saving scheme
    mesh = mshr.generate_mesh(DOMAIN, fem_res)
    V = fe.FunctionSpace(mesh, "CG", 1)
    dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

    u_sols = []
    u_sols_evaluated = np.zeros((mc_samples, len(dof_coordinates)))
    xis = np.zeros((mc_samples, randomFieldV.J + 2))
    for i in range(mc_samples):
        print(f"Iteration {i+1}/{mc_samples}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), randomFieldV.J)
        F = np.random.random(size=2)
        xis[i] = np.concatenate((xi, F))
        u_sols.append(solve_poisson_for_given_sample_rhs_random(fem_res, jacobianV, xi, F))
        for j, point_coords in enumerate(dof_coordinates):
            u_sols_evaluated[i, j] = u_sols[i](fe.Point(point_coords))
    poisson_rhs_random_save_results_to_csv(u_sols_evaluated, xis, fem_res, kl_res)

def poisson_rhs_random_analyse_two_resolutions_from_data_u_hat(resolution_sparse, resolution_fine, P_hat):
    # Here the sparse resolution is used for both fem_res and the fine resolution is used just for the kl_res of the fine solution
    # So for both functionspaces the sparse mesh resolution is used for the mesh
    data_sparse = np.genfromtxt(f'poisson_rhs_random_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    data_fine = np.genfromtxt(f'poisson_rhs_random_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')

    mesh_sparse = mshr.generate_mesh(DOMAIN, resolution_sparse)
    V_sparse = fe.FunctionSpace(mesh_sparse, "CG", 1)

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [4]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= data_sparse.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------


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

        # # Plot sparse solution

        # c = fe.plot(mean_sol_sparse, title=f"Mean sol sparse resolution: {resolution_sparse}, MC Samples: {mc_sample_size}")
        # plt.scatter(P_hat.x(), P_hat.y(), color='blue', label=r'$\hat{P}$', s=4)
        # plt.legend()
        # plt.colorbar(c)
        # plt.show()

    # Plot fine solution
    x_coords = mesh_sparse.coordinates()[:, 0]
    y_coords = mesh_sparse.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]
    z_values_fine_mean = []
    z_values_fine_var = []

    for i in range(len(x_coords)):
        z_values_fine_mean.append(mean_sol_fine(x_coords[i], y_coords[i]))
        z_values_fine_var.append(var_sol_fine(x_coords[i], y_coords[i]))

    grid_z = griddata((x_coords, y_coords), z_values_fine_mean, (grid_x, grid_y), method='linear')
    fig_mean = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
    fig_mean.update_layout(title=dict(text="Mean estimation û(x,y) rhs random", x=0.5, y=0.95),
                    autosize=True,
                    # height=400,
                    margin=dict(l=10, r=10, b=10, t=20),
                    scene=dict(
                        xaxis_title='x-axis',
                        yaxis_title='y-axis',
                        zaxis_title='û(x, y)'))
    fig_mean.show()

    grid_z = griddata((x_coords, y_coords), z_values_fine_var, (grid_x, grid_y), method='linear')
    fig_var = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis', colorbar=dict(exponentformat='e'))])
    fig_var.update_layout(title=dict(text="Variance estimation û(x,y) rhs random", x=0.5, y=0.95),
                      autosize=True,
                    # height=400,
                    margin=dict(l=10, r=10, b=10, t=20),
                    scene=dict(
                        xaxis=dict(title='x-axis', exponentformat='e'),
                        yaxis=dict(title='y-axis', exponentformat='e'),
                        zaxis=dict(title='z-axis', exponentformat='e')))
    fig_var.show()


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # Plot means
    ax1.plot(mc_sample_sizes, u_sols_sparse_P_hat_means, 'bo', marker='x', linestyle='None', label='Means')
    ax1.fill_between(mc_sample_sizes, lower_confidence_bounds, upper_confidence_bounds, alpha=0.2, label='95% Confidence Interval')
    ax1.axhline(y=mean_sol_fine(P_hat), color='r', linestyle='-', label='Mean reference solution')
    ax1.fill_between(mc_sample_sizes, lower_confidence_bound_fine, upper_confidence_bound_fine, alpha=0.2, color='red', label='95% Confidence Interval reference solution')
    ax1.set_xscale('log')
    ax1.set_xlabel('MC Samples')
    ax1.set_ylabel('Means')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot variance
    ax2.plot(mc_sample_sizes, u_sols_sparse_P_hat_vars, 'go', marker='x', linestyle='None', label='Variance')
    ax2.axhline(y=var_sol_fine(P_hat), color='r', linestyle='-', label='Variance reference solution')
    ax2.set_xscale('log')
    ax2.set_xlabel('MC Samples')
    ax2.set_ylabel('Variance')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle(f'Means and Variance of û(x,y) in point ({P_hat.x()}, {P_hat.y()}) rhs random')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot L2 and H1 errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(mc_sample_sizes, L2_errors, 'bo', marker='x', label='L2 Error')
    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('MC Samples')
    ax1.set_ylabel('L2 Error')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(mc_sample_sizes, H1_errors, 'bo', marker='x', label='H1 Error')
    ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.set_xlabel('MC Samples')
    ax2.set_ylabel('H1 Error')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('L2 and H1 Errors of û(x,y) to reference solution rhs random')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Sobol

def poisson_rhs_random_sobol_run_samples_and_save(mc_sample_size: int, fem_res: int, kl_res: int, size_of_xi: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None) -> None:

    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)
    
    # Collect points P
    N = len(randomFieldV.basis_functions)
    
    # Sample all xis, not only the ones that are investigated
    len_xi_total = randomFieldV.J + 2
    A = np.zeros((mc_sample_size, len_xi_total))
    B = np.zeros((mc_sample_size, len_xi_total))

    for xi_index in range(randomFieldV.J):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for F_index in range(2):
        A[:, randomFieldV.J + F_index] = np.random.random(size=mc_sample_size)
        B[:, randomFieldV.J + F_index] = np.random.random(size=mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        print(f"First loop Iteration {m+1}/{mc_sample_size}")
        u_sol_A = solve_poisson_for_given_sample_rhs_random(fem_res, jacobianV, A[m, :randomFieldV.J], A[m, randomFieldV.J:])
        u_sol_B = solve_poisson_for_given_sample_rhs_random(fem_res, jacobianV, B[m, :randomFieldV.J], B[m, randomFieldV.J:])
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

    for i in range(size_of_xi + 2):
        C_i = np.zeros((mc_sample_size, len_xi_total))
        for xi_index in range(randomFieldV.J):
            if xi_index == i and xi_index < size_of_xi:
                C_i[:, xi_index] = A[:, xi_index]
            else:
                C_i[:, xi_index] = B[:, xi_index]

        for F_index in range(2):
            if size_of_xi + F_index == i:
                C_i[:, randomFieldV.J + F_index] = A[:, randomFieldV.J + F_index]
            else:
                C_i[:, randomFieldV.J + F_index] = B[:, randomFieldV.J + F_index]
        
        f_C_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            print(f"Second loop xi {i+1} of {size_of_xi}, Iteration {m+1}/{mc_sample_size}")
            u_sol_C_i = solve_poisson_for_given_sample_rhs_random(fem_res, jacobianV, C_i[m, :randomFieldV.J], C_i[m, randomFieldV.J:])
            for basis_function_index, basis_function in enumerate(randomFieldV.basis_functions):
                transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
                active_quad_points = QUAD_POINTS_2DD_6
                quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
                int_C_i = 0
                for quad_point in quad_points:
                    int_C_i += u_sol_C_i(quad_point.point) * quad_point.weight
                f_C_i[m, basis_function_index] = int_C_i
        f_C_is.append(f_C_i)
        
    base_path = f'sobol_data_storage/poisson_rhs_random/femres_{fem_res}_klres_{kl_res}_size_of_xi_{size_of_xi}'
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

    for i in range(size_of_xi + 2):
        f_C_i_path = os.path.join(base_path, f'f_C_{i}.npy')
        if os.path.exists(f_C_i_path):
            f_C_i_existing = np.load(f_C_i_path)
            f_C_is[i] = np.concatenate((f_C_i_existing, f_C_is[i]), axis=0)
        np.save(f_C_i_path, f_C_is[i])

    print("Sobol data saved")

def poisson_rhs_random_sobol_calc_indices_from_data(fem_res: int, kl_res: int, size_of_xi: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None) -> tuple[np.array, np.array, int]:

    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    base_path = f'sobol_data_storage/poisson_rhs_random/femres_{fem_res}_klres_{kl_res}_size_of_xi_{size_of_xi}'
    f_A_path = os.path.join(base_path, 'f_A.npy')
    f_B_path = os.path.join(base_path, 'f_B.npy')

    f_A = np.load(f_A_path)
    f_B = np.load(f_B_path)

    S_single = np.zeros(size_of_xi + 2)
    S_total = np.zeros(size_of_xi + 2)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions]

    for i in range(size_of_xi + 2):
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

def poisson_rhs_random_plot_sobols(S_single, S_total, mc_sample_size, title):

    fig, ax = plt.subplots(figsize=(10, 5))

    # Set width for each bar
    bar_width = 0.35

    ax.bar(np.arange(len(S_single)), S_single, width=bar_width, label='First Order')
    ax.bar(np.arange(len(S_single)) + bar_width, S_total, width=bar_width, label='Total Effect')
    x_labels = [fr"$\xi_{{{i+1}}}$" for i in range(len(S_single) - 2)] + [r"$F_1$", r"$F_2$"]
    ax.set_xticklabels(x_labels)
    ax.set_xticks(np.arange(len(S_single)) + bar_width / 2)
    ax.set_ylabel('Sensitivity [-]')
    ax.set_title(f'{title} Sample Size: {mc_sample_size}')
    ax.grid(True)
    ax.legend()
    plt.show()


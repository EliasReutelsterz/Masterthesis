import matplotlib.pyplot as plt
import fenics as fe
import mshr
import numpy as np
from scipy.linalg import eig
import sympy as sp
from scipy.interpolate import griddata
import plotly.graph_objects as go
import time
import os
import plotly.io as pio
pio.renderers.default = 'notebook'

# Poisson

def analyse_two_resolutions_from_data_u(resolution_sparse: int, resolution_fine: int, P_hat, len_xi: int, randomFieldV_sparse: RandomFieldV = None, randomFieldV_fine: RandomFieldV = None):
    # Here the sparse resolution is used for both fem_res and the fine resolution is used just for the kl_res of the fine solution
    # So for both functionspaces the sparse mesh resolution is used for the mesh
    data_sparse = np.genfromtxt(f'poisson_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    xis_sparse = np.genfromtxt(f'poisson_sample_storage/xis_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    data_fine = np.genfromtxt(f'poisson_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')
    xis_fine = np.genfromtxt(f'poisson_sample_storage/xis_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')

    if not randomFieldV_sparse:
        randomFieldV_sparse, _ = calculate_vector_field_eigenpairs(resolution_sparse)
    if not randomFieldV_fine:
        randomFieldV_fine, _ = calculate_vector_field_eigenpairs(resolution_fine)

    mesh_sparse = mshr.generate_mesh(DOMAIN, resolution_sparse)

    V_sparse = fe.FunctionSpace(mesh_sparse, "CG", 1)

    NVA_sparse = non_varying_area(len_xi, randomFieldV_sparse)
    NVA_fine = non_varying_area(len_xi, randomFieldV_fine)

    if not pointInNVA(P_hat, NVA_sparse) or not pointInNVA(P_hat, NVA_fine):
        raise ValueError("Point P is not in the non-varying area")
    if NVA_sparse.radius() < NVA_fine.radius():
        mesh_NVA = mshr.generate_mesh(NVA_sparse, resolution_sparse)
    else:
        mesh_NVA = mshr.generate_mesh(NVA_fine, resolution_sparse)
    V_NVA = fe.FunctionSpace(mesh_NVA, "CG", 1)
    dof_coordinates = V_NVA.tabulate_dof_coordinates().reshape((-1, mesh_NVA.geometry().dim()))

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [4]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= data_sparse.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------


    ## Calculation ##

    mean_sol_fine = fe.Function(V_NVA)
    mean_sol_fine.set_allow_extrapolation(True)
    mean_sol_fine.vector()[:] = 0

    var_sol_fine = fe.Function(V_NVA)
    var_sol_fine.set_allow_extrapolation(True)
    var_sol_fine.vector()[:] = 0

    u_sols_fine = []
    
    # Mean fine solution
    for data_index, data_row in enumerate(data_fine):
        u_sol = fe.Function(V_sparse)
        u_sol.set_allow_extrapolation(True)
        u_sol.vector()[:] = data_row
        u_sols_fine.append(u_sol)
        for point_index, point_coords in enumerate(dof_coordinates):
            point = inverse_mapping(fe.Point(point_coords), randomFieldV_fine, xis_fine[data_index], resolution_fine)
            mean_sol_fine.vector()[point_index] += u_sols_fine[data_index](point) / data_fine.shape[0]

    # Variance fine solution
    for data_index, data_row in enumerate(data_fine):
        for point_index, point_coords in enumerate(dof_coordinates):
            point = inverse_mapping(fe.Point(point_coords), randomFieldV_fine, xis_fine[data_index], resolution_fine)
            var_sol_fine.vector()[point_index] += (u_sols_fine[data_index](point) - mean_sol_fine(point))**2 / data_fine.shape[0]
    
    lower_confidence_bound_fine = mean_sol_fine(P_hat) - 1.96 * np.sqrt(var_sol_fine(P_hat) / data_fine.shape[0])
    upper_confidence_bound_fine = mean_sol_fine(P_hat) + 1.96 * np.sqrt(var_sol_fine(P_hat) / data_fine.shape[0])

    u_sols_sparse_P_hat_means = []
    u_sols_sparse_P_hat_vars = []
    lower_confidence_bounds = []
    upper_confidence_bounds = []
    L2_errors = []
    H1_errors = []
    for mc_sample_index, mc_sample_size in enumerate(mc_sample_sizes):

        print(f"MC Sample Size: {mc_sample_size} from total {np.sum(mc_sample_sizes)}")

        mean_sol_sparse = fe.Function(V_NVA)
        mean_sol_sparse.set_allow_extrapolation(True)
        mean_sol_sparse.vector()[:] = 0

        var_sol_sparse = fe.Function(V_NVA)
        var_sol_sparse.set_allow_extrapolation(True)
        var_sol_sparse.vector()[:] = 0

        u_sols_sparse = []

        # get data split
        if mc_sample_index == 0:
            data_sparse_samples = data_sparse[:mc_sample_size]
            data_sparse_xis = xis_sparse[:mc_sample_size]
        else:
            data_sparse_samples = data_sparse[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]
            data_sparse_xis = xis_sparse[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]

        # Mean sparse solution
        for data_index, data_row in enumerate(data_sparse_samples):
            print(f"Mean calc, MC Sample Size: {mc_sample_size}, Data Index: {data_index}")
            u_sol = fe.Function(V_sparse)
            u_sol.set_allow_extrapolation(True)
            u_sol.vector()[:] = data_row
            u_sols_sparse.append(u_sol)
            for point_index, point_coords in enumerate(dof_coordinates):
                point = inverse_mapping(fe.Point(point_coords), randomFieldV_sparse, data_sparse_xis[data_index], resolution_sparse)
                mean_sol_sparse.vector()[point_index] += u_sols_sparse[data_index](point) / mc_sample_size
        u_sols_sparse_P_hat_means.append(mean_sol_sparse(P_hat))

        # Variance sparse solution
        for data_index, data_row in enumerate(data_sparse_samples):
            print(f"Var calc, MC Sample Size: {mc_sample_size}, Data Index: {data_index}")
            for point_index, point_coords in enumerate(dof_coordinates):
                point = inverse_mapping(fe.Point(point_coords), randomFieldV_sparse, data_sparse_xis[data_index], resolution_sparse)
                var_sol_sparse.vector()[point_index] += (u_sols_sparse[data_index](point) - mean_sol_sparse(fe.Point(point_coords)))**2 / mc_sample_size
        u_sols_sparse_P_hat_vars.append(var_sol_sparse(P_hat))


        # Calculate confidence intervals
        lower_confidence_bounds.append(u_sols_sparse_P_hat_means[-1] - 1.96 * np.sqrt(u_sols_sparse_P_hat_vars[-1] / mc_sample_size))
        upper_confidence_bounds.append(u_sols_sparse_P_hat_means[-1] + 1.96 * np.sqrt(u_sols_sparse_P_hat_vars[-1] / mc_sample_size))
        
        # Calculate L2 and H1 errors
        L2_errors.append(fe.errornorm(mean_sol_fine, mean_sol_sparse, 'L2'))
        H1_errors.append(fe.errornorm(mean_sol_fine, mean_sol_sparse, 'H1'))


    ## Plots ##

    # Plot fine solution
    x_coords = mesh_NVA.coordinates()[:, 0]
    y_coords = mesh_NVA.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]
    z_values_fine_mean = []
    z_values_fine_var = []

    for i in range(len(x_coords)):
        z_values_fine_mean.append(mean_sol_fine(x_coords[i], y_coords[i]))
        z_values_fine_var.append(var_sol_fine(x_coords[i], y_coords[i]))

    grid_z = griddata((x_coords, y_coords), z_values_fine_mean, (grid_x, grid_y), method='linear')
    fig_mean = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
    fig_mean.update_layout(title=dict(text="Mean estimation u(x,y)", x=0.5, y=0.95),
                    autosize=True,
                    # height=400,
                    margin=dict(l=10, r=10, b=10, t=20),
                    scene=dict(
                        xaxis_title='x-axis',
                        yaxis_title='y-axis',
                        zaxis_title='u(x, y)'))
    fig_mean.show()

    grid_z = griddata((x_coords, y_coords), z_values_fine_var, (grid_x, grid_y), method='linear')
    fig_var = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis', colorbar=dict(exponentformat='e'))])
    fig_var.update_layout(title=dict(text="Variance estimation u(x,y)", x=0.5, y=0.95),
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

    plt.suptitle(f'Means and Variance of u(x,y) in point ({P_hat.x()}, {P_hat.y()})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot L2 and H1 errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(mc_sample_sizes, L2_errors, 'bo', marker='x', label='L2 Error')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('MC Samples')
    ax1.set_ylabel('L2 Error')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(mc_sample_sizes, H1_errors, 'bo', marker='x', label='H1 Error')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('MC Samples')
    ax2.set_ylabel('H1 Error')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('L2 and H1 Errors of u(x,y) on NVA to reference solution')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def poisson_func_valued_sobol_calculate_A_u(mc_sample_size: int, mesh_resolution: int, size_of_xi: int, NVA_radius: float):

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    NVA = mshr.Circle(CENTER, NVA_radius)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)

    # Sample
    A = np.zeros((mc_sample_size, size_of_xi))

    for xi_index in range(size_of_xi):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_A = solve_poisson_for_given_sample(mesh_resolution, jacobianV, A[m], RHS_F)
        for P_index, P in enumerate(Ps):
            P_A = inverse_mapping(P, randomFieldV, A[m], mesh_resolution)
            f_A[m, P_index] = u_sol_A(P_A)

    np.save('sobol_data_storage/poisson/u/A.npy', A)
    np.save('sobol_data_storage/poisson/u/f_A.npy', f_A)

def poisson_func_valued_sobol_calculate_B_u(mc_sample_size: int, mesh_resolution: int, size_of_xi: int, NVA_radius: float):

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    NVA = mshr.Circle(CENTER, NVA_radius)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)

    # Sample
    B = np.zeros((mc_sample_size, size_of_xi))

    for xi_index in range(size_of_xi):
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_B = solve_poisson_for_given_sample(mesh_resolution, jacobianV, B[m], RHS_F)
        for P_index, P in enumerate(Ps):
            P_A = inverse_mapping(P, randomFieldV, B[m], mesh_resolution)
            f_B[m, P_index] = u_sol_B(P_A)

    np.save('sobol_data_storage/poisson/u/B.npy', B)
    np.save('sobol_data_storage/poisson/u/f_B.npy', f_B)

def poisson_calc_one_sobol_index_u(i: int, mc_sample_size: int, mesh_resolution: int, size_of_xi: int, NVA_radius: float) -> tuple[float, float]:
    
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    # Collect points P
    NVA = mshr.Circle(CENTER, NVA_radius)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    
    # Sample
    A = np.load('sobol_data_storage/poisson/u/A.npy')
    B = np.load('sobol_data_storage/poisson/u/B.npy')

    f_A = np.load('sobol_data_storage/poisson/u/f_A.npy')
    f_B = np.load('sobol_data_storage/poisson/u/f_B.npy')

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    A_B_i = np.zeros((mc_sample_size, size_of_xi))
    for param_index in range(size_of_xi):
        if param_index == i:
            A_B_i[:, param_index] = B[:, param_index]
        else:
            A_B_i[:, param_index] = A[:, param_index]
    
    f_A_B_i = np.zeros((mc_sample_size, N))
    for m in range(mc_sample_size):
        u_sol_A_B_i = solve_poisson_for_given_sample(mesh_resolution, jacobianV, A_B_i[m], RHS_F)
        for P_index, P in enumerate(Ps):
            P_A_B_i = inverse_mapping(P, randomFieldV, A_B_i[m], mesh_resolution)
            f_A_B_i[m, P_index] = u_sol_A_B_i(P_A_B_i)
    
    var_g_i = (1 / mc_sample_size) * np.sum(f_B * (f_A_B_i - f_A), axis=0)
    E_var_g_i = (1 / mc_sample_size) * np.sum(f_A * (f_A - f_A_B_i), axis=0)
    var_g = (1 / mc_sample_size) * np.sum(f_A**2, axis=0) - f_0_squared

    numerator_s_single = np.sum(weights * var_g_i)
    numerator_s_total = np.sum(weights * E_var_g_i)
    denominator = np.sum(weights * var_g)

    S_single = numerator_s_single / denominator
    S_total = numerator_s_total / denominator
    return S_single, S_total

def inverse_mapping_visualisation(mesh_resolution, randomFieldV, jacobianV):

    P = fe.Point(0.01, 0.01)
    len_xi = 2

    xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), len_xi)

    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    # perturbed mesh based on the "original" mesh used for the KL-expansion
    perturbed_coordinates = mesh.coordinates().copy()
    for index, coordinate in enumerate(mesh.coordinates()):
        perturbed_coordinates[index] = randomFieldV(coordinate, xi)
    # Create a new mesh with the perturbed coordinates
    perturbed_mesh = fe.Mesh(mesh)
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    # Find the cell containing point P
    image_tria_coords = None
    for cell in fe.cells(perturbed_mesh):
        if cell.contains(P):
            image_tria_coords = np.array(cell.get_vertex_coordinates()).reshape((-1, 2))
            break

    if image_tria_coords is None:
        raise ValueError("Point P is not inside any cell of the perturbed mesh")
            
    image_bary_coords = barycentric_coords(P, image_tria_coords)

    print(f"image_bary_coords: {image_bary_coords}")

    perturbed_coords = perturbed_mesh.coordinates()
    original_coords = mesh.coordinates()

    indices = []
    for vertex in image_tria_coords:
        for i, coord in enumerate(perturbed_coords):
            if np.allclose(vertex, coord):
                indices.append(i)
                break

    # Get the corresponding coordinates in the original mesh
    original_tria_coords = original_coords[indices]
    print(f"image_tria_coords: {image_tria_coords}")
    print(f"original_tria_coords: {original_tria_coords}")

    P_hat = (
        image_bary_coords[0] * original_tria_coords[0] +
        image_bary_coords[1] * original_tria_coords[1] +
        image_bary_coords[2] * original_tria_coords[2]
    )
    print(f"New inner point in the original triangle: {P_hat}")

    # Plot the original triangle and the new_inner_point
    plt.figure()

    # Plot original triangle
    plt.plot(*zip(*original_tria_coords, original_tria_coords[0]), 'bo-', label='Original Triangle')
    plt.scatter(*P_hat, color='blue', label='Inverse Mapped P')

    # Plot perturbed triangle
    plt.plot(*zip(*image_tria_coords, image_tria_coords[0]), 'ro-', label='Perturbed Triangle')
    plt.scatter(P.x(), P.y(), color='red', label='Point P')

    plt.legend()
    plt.title('Original and Perturbed Triangles with Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Calculate the solution at the inverse mapped point
    mesh_resolution_solution = 3
    u_sol = solve_poisson_for_given_sample(mesh_resolution_solution, jacobianV, xi, RHS_F)
    inverse_mapped_P_solution = u_sol(P_hat)
    print(f"inverse_mapped_P_solution: {inverse_mapped_P_solution}")

def non_varying_area(len_xi, randomFieldV: RandomFieldV):
    # Only valid for piecewise constant basis functions
    R_nv = np.sqrt(3) * np.sqrt((np.sum([np.sqrt(randomFieldV.eigenvalues[m]) * np.max([np.abs(randomFieldV.eigenvectors[j, m]) for j in range(randomFieldV.N)]) for m in range(len_xi)]))**2 \
                                + (np.sum([np.sqrt(randomFieldV.eigenvalues[m]) * np.max([np.abs(randomFieldV.eigenvectors[randomFieldV.N + j, m]) for j in range(randomFieldV.N)]) for m in range(len_xi)]))**2)
    print(f"R_nv (maximal perturbation distance): {R_nv}") # maximal perturbation distance

    # Non-Varying Area
    if R_nv >= 1:
            raise ValueError(f"for len(xi): {len_xi} the non-varying area is the empty set")
    return mshr.Circle(CENTER, 1 - R_nv)

def pointInNVA(P, NVA):
    if NVA.radius() < P.distance(CENTER):
        return False
    return True

def functional_valued_output_sobol_estimation_u(mc_sample_size: int, mesh_resolution: int, size_of_xi: int) -> tuple[np.array, np.array]:

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    # Collect points P
    NVA = non_varying_area(size_of_xi, randomFieldV)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    
    # Sample
    A = np.zeros((mc_sample_size, size_of_xi))
    B = np.zeros((mc_sample_size, size_of_xi))

    for xi_index in range(size_of_xi):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_A = solve_poisson_for_given_sample(mesh_resolution, jacobianV, A[m], RHS_F)
        u_sol_B = solve_poisson_for_given_sample(mesh_resolution, jacobianV, B[m], RHS_F)
        for P_index, P in enumerate(Ps):
            P_A = inverse_mapping(P, randomFieldV, A[m], mesh_resolution)
            f_A[m, P_index] = u_sol_A(P_A)
            P_B = inverse_mapping(P, randomFieldV, B[m], mesh_resolution)
            f_B[m, P_index] = u_sol_B(P_B)
        
    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    S_single = np.zeros(size_of_xi)
    S_total = np.zeros(size_of_xi)

    for i in range(size_of_xi):
        A_B_i = np.zeros((mc_sample_size, size_of_xi))
        for param_index in range(size_of_xi):
            if param_index == i:
                A_B_i[:, param_index] = B[:, param_index]
            else:
                A_B_i[:, param_index] = A[:, param_index]
        
        f_A_B_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            u_sol_A_B_i = solve_poisson_for_given_sample(mesh_resolution, jacobianV, A_B_i[m], RHS_F)
            for P_index, P in enumerate(Ps):
                P_A_B_i = inverse_mapping(P, randomFieldV, A_B_i[m], mesh_resolution)
                f_A_B_i[m, P_index] = u_sol_A_B_i(P_A_B_i)
        
        var_g_i = (1 / mc_sample_size) * np.sum(f_B * (f_A_B_i - f_A), axis=0)
        E_var_g_i = (1 / mc_sample_size) * np.sum(f_A * (f_A - f_A_B_i), axis=0)
        var_g = (1 / mc_sample_size) * np.sum(f_A**2, axis=0) - f_0_squared

        numerator_s_single = np.sum(weights * var_g_i)
        numerator_s_total = np.sum(weights * E_var_g_i)
        denominator = np.sum(weights * var_g)

        S_single[i] = numerator_s_single / denominator
        S_total[i] = numerator_s_total / denominator

    return S_single, S_total

def compare_all_sobols_single_point_evaluation(mc_sample_size: int, mesh_resolution: int, size_of_xi: int, P_coords:list) -> tuple[np.array, np.array]:

    #! Currently not used because global calculation already implemented

    # consider sampling all xi and not only the ones that are investigated
    

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)

    P = fe.Point(np.array(P_coords))

    A = np.zeros((mc_sample_size, size_of_xi))
    B = np.zeros((mc_sample_size, size_of_xi))
    for xi_index in range(size_of_xi):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
    f_A = np.zeros(mc_sample_size)
    f_B = np.zeros(mc_sample_size)
    for m in range(mc_sample_size):
        P_A = inverse_mapping(P, randomFieldV, A[m], mesh_resolution)
        f_A[m] = solve_poisson_for_given_sample(mesh_resolution, jacobianV, A[m], RHS_F)(P_A)
        P_B = inverse_mapping(P, randomFieldV, B[m], mesh_resolution)
        f_B[m] = solve_poisson_for_given_sample(mesh_resolution, jacobianV, B[m], RHS_F)(P_B)
    
    f_0_squared = np.mean(f_A) * np.mean(f_B)

    S_single = np.zeros(size_of_xi)
    S_total = np.zeros(size_of_xi)
    
    for i in range(size_of_xi):
        A_B_i = np.zeros((mc_sample_size, size_of_xi))
        for param_index in range(size_of_xi):
            if param_index == i:
                A_B_i[:, param_index] = B[:, param_index]
            else:
                A_B_i[:, param_index] = A[:, param_index]
        
        f_A_B_i = np.zeros(mc_sample_size)
        for m in range(mc_sample_size):
            P_A_B_i = inverse_mapping(P, randomFieldV, A_B_i[m], mesh_resolution)
            f_A_B_i[m] = solve_poisson_for_given_sample(mesh_resolution, jacobianV, A_B_i[m], RHS_F)(P_A_B_i)
        
        denominator = 1/mc_sample_size * np.sum([f_A[j]**2 for j in range(mc_sample_size)]) - f_0_squared
        S_single[i] = (1/mc_sample_size * np.sum([f_B[j] * (f_A_B_i[j] - f_A[j]) for j in range(mc_sample_size)])) / denominator
        S_total[i] = (1/mc_sample_size * np.sum([f_A[j] * (f_A[j] - f_A_B_i[j]) for j in range(mc_sample_size)])) / denominator

    return S_single, S_total


# Poisson rhs random

def poisson_rhs_random_analyse_two_resolutions_from_data_u(resolution_sparse: int, resolution_fine: int, P_hat, len_xi: int, randomFieldV_sparse: RandomFieldV = None, randomFieldV_fine: RandomFieldV = None):
    # Here the sparse resolution is used for both fem_res and the fine resolution is used just for the kl_res of the fine solution
    # So for both functionspaces the sparse mesh resolution is used for the mesh
    data_sparse = np.genfromtxt(f'poisson_rhs_random_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    xis_sparse = np.genfromtxt(f'poisson_rhs_random_sample_storage/xis_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    data_fine = np.genfromtxt(f'poisson_rhs_random_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')
    xis_fine = np.genfromtxt(f'poisson_rhs_random_sample_storage/xis_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')

    if not randomFieldV_sparse:
        randomFieldV_sparse, _ = calculate_vector_field_eigenpairs(resolution_sparse)
    if not randomFieldV_fine:
        randomFieldV_fine, _ = calculate_vector_field_eigenpairs(resolution_fine)

    mesh_sparse = mshr.generate_mesh(DOMAIN, resolution_sparse)
    mesh_fine = mshr.generate_mesh(DOMAIN, resolution_sparse)

    V_sparse = fe.FunctionSpace(mesh_sparse, "CG", 1)

    NVA_sparse = non_varying_area(len_xi, randomFieldV_sparse)
    NVA_fine = non_varying_area(len_xi, randomFieldV_fine)

    if not pointInNVA(P_hat, NVA_sparse) or not pointInNVA(P_hat, NVA_fine):
        raise ValueError("Point P is not in the non-varying area")
    if NVA_sparse.radius() < NVA_fine.radius():
        mesh_NVA = mshr.generate_mesh(NVA_sparse, resolution_sparse)
    else:
        mesh_NVA = mshr.generate_mesh(NVA_fine, resolution_sparse)
    V_NVA = fe.FunctionSpace(mesh_NVA, "CG", 1)
    dof_coordinates = V_NVA.tabulate_dof_coordinates().reshape((-1, mesh_NVA.geometry().dim()))

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [4]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= data_sparse.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------


    ## Calculation ##

    mean_sol_fine = fe.Function(V_NVA)
    mean_sol_fine.set_allow_extrapolation(True)
    mean_sol_fine.vector()[:] = 0

    var_sol_fine = fe.Function(V_NVA)
    var_sol_fine.set_allow_extrapolation(True)
    var_sol_fine.vector()[:] = 0

    u_sols_fine = []
    
    # Mean fine solution
    for data_index, data_row in enumerate(data_fine):
        u_sol = fe.Function(V_sparse)
        u_sol.set_allow_extrapolation(True)
        u_sol.vector()[:] = data_row
        u_sols_fine.append(u_sol)
        for point_index, point_coords in enumerate(dof_coordinates):
            point = inverse_mapping(fe.Point(point_coords), randomFieldV_fine, xis_fine[data_index, :-2], resolution_fine)
            mean_sol_fine.vector()[point_index] += u_sols_fine[data_index](point) / data_fine.shape[0]

    # Variance fine solution
    for data_index, data_row in enumerate(data_fine):
        for point_index, point_coords in enumerate(dof_coordinates):
            point = inverse_mapping(fe.Point(point_coords), randomFieldV_fine, xis_fine[data_index, :-2], resolution_fine)
            var_sol_fine.vector()[point_index] += (u_sols_fine[data_index](point) - mean_sol_fine(point))**2 / data_fine.shape[0]
    
    lower_confidence_bound_fine = mean_sol_fine(P_hat) - 1.96 * np.sqrt(var_sol_fine(P_hat) / data_fine.shape[0])
    upper_confidence_bound_fine = mean_sol_fine(P_hat) + 1.96 * np.sqrt(var_sol_fine(P_hat) / data_fine.shape[0])

    u_sols_sparse_P_hat_means = []
    u_sols_sparse_P_hat_vars = []
    lower_confidence_bounds = []
    upper_confidence_bounds = []
    L2_errors = []
    H1_errors = []
    for mc_sample_index, mc_sample_size in enumerate(mc_sample_sizes):
        mean_sol_sparse = fe.Function(V_NVA)
        mean_sol_sparse.set_allow_extrapolation(True)
        mean_sol_sparse.vector()[:] = 0

        var_sol_sparse = fe.Function(V_NVA)
        var_sol_sparse.set_allow_extrapolation(True)
        var_sol_sparse.vector()[:] = 0

        u_sols_sparse = []

        # get data split
        if mc_sample_index == 0:
            data_sparse_samples = data_sparse[:mc_sample_size]
            data_sparse_xis = xis_sparse[:mc_sample_size]
        else:
            data_sparse_samples = data_sparse[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]
            data_sparse_xis = xis_sparse[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]

        # Mean sparse solution
        for data_index, data_row in enumerate(data_sparse_samples):
            u_sol = fe.Function(V_sparse)
            u_sol.set_allow_extrapolation(True)
            u_sol.vector()[:] = data_row
            u_sols_sparse.append(u_sol)
            for point_index, point_coords in enumerate(dof_coordinates):
                point = inverse_mapping(fe.Point(point_coords), randomFieldV_sparse, data_sparse_xis[data_index, :-2], resolution_sparse)
                mean_sol_sparse.vector()[point_index] += u_sols_sparse[data_index](point) / mc_sample_size
        u_sols_sparse_P_hat_means.append(mean_sol_sparse(P_hat))

        # Variance sparse solution
        for data_index, data_row in enumerate(data_sparse_samples):
            for point_index, point_coords in enumerate(dof_coordinates):
                point = inverse_mapping(fe.Point(point_coords), randomFieldV_sparse, data_sparse_xis[data_index, :-2], resolution_sparse)
                var_sol_sparse.vector()[point_index] += (u_sols_sparse[data_index](point) - mean_sol_sparse(fe.Point(point_coords)))**2 / mc_sample_size
        u_sols_sparse_P_hat_vars.append(var_sol_sparse(P_hat))


        # Calculate confidence intervals
        lower_confidence_bounds.append(u_sols_sparse_P_hat_means[-1] - 1.96 * np.sqrt(u_sols_sparse_P_hat_vars[-1] / mc_sample_size))
        upper_confidence_bounds.append(u_sols_sparse_P_hat_means[-1] + 1.96 * np.sqrt(u_sols_sparse_P_hat_vars[-1] / mc_sample_size))
        
        # Calculate L2 and H1 errors
        L2_errors.append(fe.errornorm(mean_sol_fine, mean_sol_sparse, 'L2'))
        H1_errors.append(fe.errornorm(mean_sol_fine, mean_sol_sparse, 'H1'))


    ## Plots ##

    # Plot fine solution
    x_coords = mesh_NVA.coordinates()[:, 0]
    y_coords = mesh_NVA.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[-1:1:500j, -1:1:500j]
    z_values_fine_mean = []
    z_values_fine_var = []

    for i in range(len(x_coords)):
        z_values_fine_mean.append(mean_sol_fine(x_coords[i], y_coords[i]))
        z_values_fine_var.append(var_sol_fine(x_coords[i], y_coords[i]))

    grid_z = griddata((x_coords, y_coords), z_values_fine_mean, (grid_x, grid_y), method='linear')
    fig = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis')])
    fig.update_layout(title=dict(text="Mean estimation u(x,y) rhs random", x=0.5, y=0.95),
                    autosize=True,
                    # height=400,
                    margin=dict(l=10, r=10, b=10, t=20),
                    scene=dict(
                        xaxis_title='x-axis',
                        yaxis_title='y-axis',
                        zaxis_title='u(x, y)'))
    fig.show()

    grid_z = griddata((x_coords, y_coords), z_values_fine_var, (grid_x, grid_y), method='linear')
    fig = go.Figure(data=[go.Surface(z=grid_z, x=grid_x, y=grid_y, colorscale='Viridis', colorbar=dict(exponentformat='e'))])
    fig.update_layout(title=dict(text="Variance estimation u(x,y) rhs random", x=0.5, y=0.95),
                      autosize=True,
                    # height=400,
                    margin=dict(l=10, r=10, b=10, t=20),
                    scene=dict(
                        xaxis=dict(title='x-axis', exponentformat='e'),
                        yaxis=dict(title='y-axis', exponentformat='e'),
                        zaxis=dict(title='z-axis', exponentformat='e')))
    fig.show()


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

    plt.suptitle(f'Means and Variance of u(x,y) in point ({P_hat.x()}, {P_hat.y()}) rhs random')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot L2 and H1 errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(mc_sample_sizes, L2_errors, 'bo', marker='x', label='L2 Error')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('MC Samples')
    ax1.set_ylabel('L2 Error')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(mc_sample_sizes, H1_errors, 'bo', marker='x', label='H1 Error')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('MC Samples')
    ax2.set_ylabel('H1 Error')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('L2 and H1 Errors of u(x,y) on NVA to reference solution rhs random')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def poisson_rhs_random_func_valued_sobol_calculate_A_u(mc_sample_size: int, mesh_resolution: int, size_of_xi: int, NVA_radius: float):

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    NVA = mshr.Circle(CENTER, NVA_radius)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)

    # Sample
    A = np.zeros((mc_sample_size, size_of_xi + 2))

    for xi_index in range(size_of_xi):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for F_index in range(2):
        A[:, size_of_xi + F_index] = np.random.random(size=mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_A = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, A[m, :size_of_xi], A[m, size_of_xi:])
        for P_index, P in enumerate(Ps):
            P_A = inverse_mapping(P, randomFieldV, A[m, :size_of_xi], mesh_resolution)
            f_A[m, P_index] = u_sol_A(P_A)

    np.save('sobol_data_storage/poisson_rhs_random/u/A.npy', A)
    np.save('sobol_data_storage/poisson_rhs_random/u/f_A.npy', f_A)

def poisson_rhs_random_func_valued_sobol_calculate_B_u(mc_sample_size: int, mesh_resolution: int, size_of_xi: int, NVA_radius: float):

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    NVA = mshr.Circle(CENTER, NVA_radius)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)

    # Sample
    B = np.zeros((mc_sample_size, size_of_xi + 2))

    for xi_index in range(size_of_xi):
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for F_index in range(2):
        B[:, size_of_xi + F_index] = np.random.random(size=mc_sample_size)

    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_B = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, B[m, :size_of_xi], B[m, size_of_xi:])
        for P_index, P in enumerate(Ps):
            P_B = inverse_mapping(P, randomFieldV, B[m, :size_of_xi], mesh_resolution)
            f_B[m, P_index] = u_sol_B(P_B)

    np.save('sobol_data_storage/poisson_rhs_random/u/B.npy', B)
    np.save('sobol_data_storage/poisson_rhs_random/u/f_B.npy', f_B)

def poisson_rhs_random_calc_one_sobol_index_u(i: int, mc_sample_size: int, mesh_resolution: int, size_of_xi: int, NVA_radius: float) -> tuple[float, float]:
    
    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    # Collect points P
    NVA = mshr.Circle(CENTER, NVA_radius)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    
    # Sample
    A = np.load('sobol_data_storage/poisson_rhs_random/u/A.npy')
    B = np.load('sobol_data_storage/poisson_rhs_random/u/B.npy')

    f_A = np.load('sobol_data_storage/poisson_rhs_random/u/f_A.npy')
    f_B = np.load('sobol_data_storage/poisson_rhs_random/u/f_B.npy')

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)


    A_B_i = np.zeros((mc_sample_size, size_of_xi + 2))
    for param_index in range(size_of_xi + 2):
        if param_index == i:
            A_B_i[:, param_index] = B[:, param_index]
        else:
            A_B_i[:, param_index] = A[:, param_index]
    
    f_A_B_i = np.zeros((mc_sample_size, N))
    for m in range(mc_sample_size):
        u_sol_A_B_i = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, A_B_i[m, :size_of_xi], A_B_i[m, size_of_xi:])
        for P_index, P in enumerate(Ps):
            P_A_B_i = inverse_mapping(P, randomFieldV, A_B_i[m, :size_of_xi], mesh_resolution)
            f_A_B_i[m, P_index] = u_sol_A_B_i(P_A_B_i)
    
    var_g_i = (1 / mc_sample_size) * np.sum(f_B * (f_A_B_i - f_A), axis=0)
    E_var_g_i = (1 / mc_sample_size) * np.sum(f_A * (f_A - f_A_B_i), axis=0)
    var_g = (1 / mc_sample_size) * np.sum(f_A**2, axis=0) - f_0_squared

    numerator_s_single = np.sum(weights * var_g_i)
    numerator_s_total = np.sum(weights * E_var_g_i)
    denominator = np.sum(weights * var_g)

    S_single = numerator_s_single / denominator
    S_total = numerator_s_total / denominator

    return S_single, S_total

def rhs_random_functional_valued_output_sobol_estimation_u(mc_sample_size: int, mesh_resolution: int, size_of_xi: int) -> tuple[np.array, np.array]:

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)
    
    # Collect points P
    NVA = non_varying_area(size_of_xi, randomFieldV)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    
    # Sample
    A = np.zeros((mc_sample_size, size_of_xi + 2))
    B = np.zeros((mc_sample_size, size_of_xi + 2))

    for xi_index in range(size_of_xi):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for F_index in range(2):
        A[:, size_of_xi + F_index] = np.random.random(size=mc_sample_size)
        B[:, size_of_xi + F_index] = np.random.random(size=mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_A = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, A[m, :size_of_xi], A[m, size_of_xi:])
        u_sol_B = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, B[m, :size_of_xi], B[m, size_of_xi:])
        for P_index, P in enumerate(Ps):
            P_A = inverse_mapping(P, randomFieldV, A[m, :size_of_xi], mesh_resolution)
            f_A[m, P_index] = u_sol_A(P_A)
            P_B = inverse_mapping(P, randomFieldV, B[m, :size_of_xi], mesh_resolution)
            f_B[m, P_index] = u_sol_B(P_B)
        
    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    S_single = np.zeros(size_of_xi + 2)
    S_total = np.zeros(size_of_xi + 2)

    for i in range(size_of_xi + 2):
        A_B_i = np.zeros((mc_sample_size, size_of_xi + 2))
        for param_index in range(size_of_xi + 2):
            if param_index == i:
                A_B_i[:, param_index] = B[:, param_index]
            else:
                A_B_i[:, param_index] = A[:, param_index]
        
        f_A_B_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            u_sol_A_B_i = solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, A_B_i[m, :size_of_xi], A_B_i[m, size_of_xi:])
            for P_index, P in enumerate(Ps):
                P_A_B_i = inverse_mapping(P, randomFieldV, A_B_i[m, :size_of_xi], mesh_resolution)
                f_A_B_i[m, P_index] = u_sol_A_B_i(P_A_B_i)
        
        var_g_i = (1 / mc_sample_size) * np.sum(f_B * (f_A_B_i - f_A), axis=0)
        E_var_g_i = (1 / mc_sample_size) * np.sum(f_A * (f_A - f_A_B_i), axis=0)
        var_g = (1 / mc_sample_size) * np.sum(f_A**2, axis=0) - f_0_squared

        numerator_s_single = np.sum(weights * var_g_i)
        numerator_s_total = np.sum(weights * E_var_g_i)
        denominator = np.sum(weights * var_g)

        S_single[i] = numerator_s_single / denominator
        S_total[i] = numerator_s_total / denominator

    return S_single, S_total

# Poisson diffusion

def diffusion_functional_valued_output_sobol_estimation_u(mc_sample_size: int, mesh_resolution: int, size_xi_v: int, size_xi_z: int) -> tuple[np.array, np.array]:

    # currently only the first size_xi_v xi are considered for calculating NVA and for sobol index calculation as well
    # for z_xi the first size_xi_z are considered for sobol calculation but the rest is used for calculation as well

    randomFieldV, jacobianV = calculate_vector_field_eigenpairs(mesh_resolution)

    randomFieldZ = z_calculate_random_field_eigenpairs(mesh_resolution, z_cov)
    
    # Collect points P
    NVA = non_varying_area(size_xi_v, randomFieldV)
    Ps = [fe.Point(basis_function.middle_point) for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    N = len(Ps)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions if pointInNVA(fe.Point(basis_function.middle_point), NVA)]
    
    size_total_xi = size_xi_v + randomFieldZ.J

    # Sample
    A = np.zeros((mc_sample_size, size_total_xi))
    B = np.zeros((mc_sample_size, size_total_xi))

    for xi_v_index in range(size_xi_v):
        A[:, xi_v_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_v_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    for xi_z_index in range(randomFieldZ.J):
        A[:, size_xi_v + xi_z_index] = np.random.normal(0, 1, mc_sample_size)
        B[:, size_xi_v + xi_z_index] = np.random.normal(0, 1, mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        u_sol_A = solve_diffusion_poisson_for_given_sample(mesh_resolution, RHS_F, randomFieldV, jacobianV, randomFieldZ, A[m, :size_xi_v], A[m, size_xi_v:])
        u_sol_B = solve_diffusion_poisson_for_given_sample(mesh_resolution, RHS_F, randomFieldV, jacobianV, randomFieldZ, B[m, :size_xi_v], B[m, size_xi_v:])
        for P_index, P in enumerate(Ps):
            P_A = inverse_mapping(P, randomFieldV, A[m, :size_xi_v], mesh_resolution)
            f_A[m, P_index] = u_sol_A(P_A)
            P_B = inverse_mapping(P, randomFieldV, B[m, :size_xi_v], mesh_resolution)
            f_B[m, P_index] = u_sol_B(P_B)
        
    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    S_single = np.zeros(size_xi_v + size_xi_z)
    S_total = np.zeros(size_xi_v + size_xi_z)

    for i in range(size_xi_v + size_xi_z):
        A_B_i = np.zeros((mc_sample_size, size_total_xi))
        for param_index in range(size_total_xi):
            if param_index == i:
                A_B_i[:, param_index] = B[:, param_index]
            else:
                A_B_i[:, param_index] = A[:, param_index]
        
        f_A_B_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            u_sol_A_B_i = solve_diffusion_poisson_for_given_sample(mesh_resolution, RHS_F, randomFieldV, jacobianV, randomFieldZ, A_B_i[m, :size_xi_v], A_B_i[m, size_xi_v:])
            for P_index, P in enumerate(Ps):
                P_A_B_i = inverse_mapping(P, randomFieldV, A_B_i[m, :size_xi_v], mesh_resolution)
                f_A_B_i[m, P_index] = u_sol_A_B_i(P_A_B_i)
        
        var_g_i = (1 / mc_sample_size) * np.sum(f_B * (f_A_B_i - f_A), axis=0)
        E_var_g_i = (1 / mc_sample_size) * np.sum(f_A * (f_A - f_A_B_i), axis=0)
        var_g = (1 / mc_sample_size) * np.sum(f_A**2, axis=0) - f_0_squared

        numerator_s_single = np.sum(weights * var_g_i)
        numerator_s_total = np.sum(weights * E_var_g_i)
        denominator = np.sum(weights * var_g)

        S_single[i] = numerator_s_single / denominator
        S_total[i] = numerator_s_total / denominator

    return S_single, S_total

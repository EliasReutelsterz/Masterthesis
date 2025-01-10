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




# Constants
CENTER = fe.Point(0, 0)
RADIUS = 1
DOMAIN = mshr.Circle(CENTER, RADIUS)
RHS_F = fe.Constant(1)
DIRICHLET_BC = fe.Constant(0)


# Define symbolic variables
x1, x2, y1, y2 = sp.symbols('x1 x2 y1 y2')

# Define symbolic covariance functions
v_cov1_1_sym = 5.0/100.0 * sp.exp(-4.0 * ((x1 - y1)**2 + (x2 - y2)**2))
v_cov1_2_sym = 1.0/100.0 * sp.exp(-0.1 * ((2*x1 - y1)**2 + (2*x2 - y2)**2))
v_cov2_1_sym = 1.0/100.0 * sp.exp(-0.1 * ((x1 - 2*y1)**2 + (x2 - 2*y2)**2))
v_cov2_2_sym = 5.0/100.0 * sp.exp(-1.0 * ((x1 - y1)**2 + (x2 - y2)**2))

# Convert symbolic functions to numerical functions
v_cov1_1 = sp.lambdify((x1, x2, y1, y2), v_cov1_1_sym, 'numpy')
v_cov1_2 = sp.lambdify((x1, x2, y1, y2), v_cov1_2_sym, 'numpy')
v_cov2_1 = sp.lambdify((x1, x2, y1, y2), v_cov2_1_sym, 'numpy')
v_cov2_2 = sp.lambdify((x1, x2, y1, y2), v_cov2_2_sym, 'numpy')

v_cov1_1_dx1_sym = sp.diff(v_cov1_1_sym, x1)
v_cov1_1_dx2_sym = sp.diff(v_cov1_1_sym, x2)
v_cov1_2_dx1_sym = sp.diff(v_cov1_2_sym, x1)
v_cov1_2_dx2_sym = sp.diff(v_cov1_2_sym, x2)
v_cov2_1_dx1_sym = sp.diff(v_cov2_1_sym, x1)
v_cov2_1_dx2_sym = sp.diff(v_cov2_1_sym, x2)
v_cov2_2_dx1_sym = sp.diff(v_cov2_2_sym, x1)
v_cov2_2_dx2_sym = sp.diff(v_cov2_2_sym, x2)

v_cov1_1_dx1 = sp.lambdify((x1, x2, y1, y2), v_cov1_1_dx1_sym, 'numpy')
v_cov1_1_dx2 = sp.lambdify((x1, x2, y1, y2), v_cov1_1_dx2_sym, 'numpy')
v_cov1_2_dx1 = sp.lambdify((x1, x2, y1, y2), v_cov1_2_dx1_sym, 'numpy')
v_cov1_2_dx2 = sp.lambdify((x1, x2, y1, y2), v_cov1_2_dx2_sym, 'numpy')
v_cov2_1_dx1 = sp.lambdify((x1, x2, y1, y2), v_cov2_1_dx1_sym, 'numpy')
v_cov2_1_dx2 = sp.lambdify((x1, x2, y1, y2), v_cov2_1_dx2_sym, 'numpy')
v_cov2_2_dx1 = sp.lambdify((x1, x2, y1, y2), v_cov2_2_dx1_sym, 'numpy')
v_cov2_2_dx2 = sp.lambdify((x1, x2, y1, y2), v_cov2_2_dx2_sym, 'numpy')


def find_affine_transformation(triangle):
    transformation_matrix = np.array([[triangle[1, 0] - triangle[0, 0], triangle[2, 0] - triangle[0, 0]],
                                    [triangle[1, 1] - triangle[0, 1], triangle[2, 1] - triangle[0, 1]]])
    transformation_vector = np.array([triangle[0, 0], triangle[0, 1]])
    return transformation_matrix, transformation_vector

class Quad_point():
    def __init__(self, point: list, weight: float):
        self.point = point
        self.weight = weight

QUAD_POINTS_2DD_5 = [Quad_point([0, 0], 3/120),
                        Quad_point([1, 0], 3/120),
                        Quad_point([0, 1], 3/120),
                        Quad_point([1/2, 0], 8/120),
                        Quad_point([1/2, 1/2], 8/120),
                        Quad_point([0, 1/2], 8/120),
                        Quad_point([1/3, 1/3], 27/120)]

QUAD_POINTS_2DD_6 = [Quad_point([(6 - np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(9 + 2 * np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(6 - np.sqrt(15)) / 21, (9 + 2 * np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(6 + np.sqrt(15)) / 21, (9 - 2 * np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([(6 + np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([(9 - 2 * np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([1 / 3, 1 / 3], 9/80)]

def triangle_area(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    # which is the determinant of the transformation matrix
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

class ConstBasisFunction():
    def __init__(self, basis_function: fe.Function, vertex_coords: np.array):
        self.function = basis_function
        self.vertex_coords = vertex_coords
        self.triangle_area = triangle_area(vertex_coords)
        self.middle_point = np.mean(vertex_coords, axis=0)

class RandomFieldV():
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array, basis_functions: list[ConstBasisFunction], N: int, J: int):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J

    def __call__(self, x, xi):
        index_supported_basis_function = self.find_supported_basis_function(x)
        return x[0] + sum([np.sqrt(self.eigenvalues[j]) * self.eigenvectors[index_supported_basis_function, j] * xi[j] for j in range(len(xi))]), \
           x[1] + sum([np.sqrt(self.eigenvalues[j]) * self.eigenvectors[self.N + index_supported_basis_function, j] * xi[j] for j in range(len(xi))])
    
    def find_supported_basis_function(self, x):
        for i, basis_function in enumerate(self.basis_functions):
            if basis_function.function(x) == 1:
                return i
        raise ValueError("No supported basis function found for x")

class DerivativeCovarianceExpression(fe.UserExpression):
    def __init__(self, f, x, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.x = x

    def eval(self, values, y):
        values[0] = self.f(self.x[0], self.x[1], y[0], y[1])

    def value_shape(self):
        return ()
    
def quad_tri(f, basis_function: ConstBasisFunction):
    transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
    active_quad_points = QUAD_POINTS_2DD_6
    quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
    integral = 0
    for quad_point in quad_points:
        integral += f(quad_point.point[0], quad_point.point[1]) * basis_function.function(quad_point.point) * quad_point.weight
    return integral

def jacob_get_c_entry(f, basis_function_l, basis_function_k):
    transformation_matrix_x, transformation_vector_x = find_affine_transformation(basis_function_l.vertex_coords)
    transformation_matrix_y, transformation_vector_y = find_affine_transformation(basis_function_k.vertex_coords)

    active_quad_points = QUAD_POINTS_2DD_6

    # triangle areas not needed it cancels out, weight just times 2
    quad_points_x = [Quad_point(np.dot(transformation_matrix_x, quad_point_x.point) + transformation_vector_x, quad_point_x.weight * 2) for quad_point_x in active_quad_points]
    quad_points_y = [Quad_point(np.dot(transformation_matrix_y, quad_point_y.point) + transformation_vector_y, quad_point_y.weight * 2) for quad_point_y in active_quad_points]
    integral = 0
     
    for quad_point_x in quad_points_x:
        for quad_point_y in quad_points_y:
            integral += f(quad_point_x.point[0], quad_point_x.point[1], quad_point_y.point[0], quad_point_y.point[1]) * quad_point_x.weight * quad_point_y.weight
    return integral

def jacob_get_C_matrices(basis_functions):
    N = len(basis_functions)
    C = []
    for k, basis_function_k in enumerate(basis_functions):
        C_k = np.zeros((4*len(basis_functions), 2))
        for l, basis_function_l in enumerate(basis_functions):
            C_k[l, 0] = jacob_get_c_entry(v_cov1_1_dx1, basis_function_l, basis_function_k)
            C_k[l, 1] = jacob_get_c_entry(v_cov1_1_dx2, basis_function_l, basis_function_k)
            C_k[l + N, 0] = jacob_get_c_entry(v_cov1_2_dx1, basis_function_l, basis_function_k)
            C_k[l + N, 1] = jacob_get_c_entry(v_cov1_2_dx2, basis_function_l, basis_function_k)
            C_k[l + 2 * N, 0] = jacob_get_c_entry(v_cov2_1_dx1, basis_function_l, basis_function_k)
            C_k[l + 2 * N, 1] = jacob_get_c_entry(v_cov2_1_dx2, basis_function_l, basis_function_k)
            C_k[l + 3 * N, 0] = jacob_get_c_entry(v_cov2_2_dx1, basis_function_l, basis_function_k)
            C_k[l + 3 * N, 1] = jacob_get_c_entry(v_cov2_2_dx2, basis_function_l, basis_function_k)
        C.append(C_k)
    return C

class JacobianV():
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array, basis_functions: list[ConstBasisFunction], N: int, J: int):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J
        self.C = jacob_get_C_matrices(self.basis_functions)
    # Alternatively implement own call function here but can create object with same inputs and then use the call function below as well

class JacobianVFixedXi():
    def __init__(self, jacobianV: JacobianV, xi: np.array):
        #! Klassenvererbung erwägen
        self.xi = xi
        self.eigenvalues = jacobianV.eigenvalues
        self.eigenvectors = jacobianV.eigenvectors
        self.basis_functions = jacobianV.basis_functions
        self.N = jacobianV.N
        self.J = jacobianV.J
        
        a_bar_T = sum([1/np.sqrt(self.eigenvalues[m]) * self.eigenvectors[:, m] * xi[m] for m in range(len(xi))]).T
        for j in range(self.N):
            a_bar_T[j] *= self.basis_functions[j].triangle_area
            a_bar_T[self.N + j] *= self.basis_functions[j].triangle_area
        zero_matrix = np.zeros(a_bar_T.shape)
        self.A = np.block([[a_bar_T, zero_matrix],
              [zero_matrix, a_bar_T]])
        
        self.C = jacobianV.C
        self.const_jacobians = [np.eye(2) + self.A @ self.C[k] for k in range(self.N)]
    
    def find_supported_basis_function(self, x):
        for i, basis_function in enumerate(self.basis_functions):
            if basis_function.function(x) == 1:
                return i
        raise ValueError("No supported basis function found for x")
    
    def __call__(self, x):
        k_hat = self.find_supported_basis_function(x)
        return self.const_jacobians[k_hat]
    
class AExpression(fe.UserExpression):
    def __init__(self, jacobianV_fixed_xi, **kwargs):
        super().__init__(**kwargs)
        self.jacobianV_fixed_xi = jacobianV_fixed_xi

    def eval(self, values, x):
        J_x = self.jacobianV_fixed_xi(x)
        inv_JTJ = np.linalg.inv(J_x.T @ J_x)
        det_J = np.linalg.det(J_x)
        A_x = inv_JTJ * det_J
        values[0] = A_x[0, 0]
        values[1] = A_x[0, 1]
        values[2] = A_x[1, 0]
        values[3] = A_x[1, 1]

    def value_shape(self):
        return (2, 2)

class detJExpression(fe.UserExpression):
    def __init__(self, jacobianV_fixed_xi, **kwargs):
        super().__init__(**kwargs)
        self.jacobianV_fixed_xi = jacobianV_fixed_xi

    def eval(self, values, x):
        J_x = self.jacobianV_fixed_xi(x)
        det_J = np.linalg.det(J_x)
        values[0] = det_J

    def value_shape(self):
        return ()

class RandomFieldExpression(fe.UserExpression):
    def __init__(self, randomfieldV: RandomFieldV, xi: np.array, **kwargs):
        super().__init__(**kwargs)
        self.randomfieldV = randomfieldV
        self.xi = xi

    def eval(self, values, x):
        values[0] = self.randomfieldV(x, self.xi)
    
    def value_shape(self):
        return ()

def get_C_entry(f, basis_function_i: ConstBasisFunction, basis_function_j: ConstBasisFunction):
    transformation_matrix_x, transformation_vector_x = find_affine_transformation(basis_function_i.vertex_coords)
    transformation_matrix_y, transformation_vector_y = find_affine_transformation(basis_function_j.vertex_coords)

    active_quad_points = QUAD_POINTS_2DD_6

    quad_points_x = [Quad_point(np.dot(transformation_matrix_x, quad_point_x.point) + transformation_vector_x, quad_point_x.weight * 2 * basis_function_i.triangle_area) for quad_point_x in active_quad_points]
    quad_points_y = [Quad_point(np.dot(transformation_matrix_y, quad_point_y.point) + transformation_vector_y, quad_point_y.weight * 2 * basis_function_j.triangle_area) for quad_point_y in active_quad_points]
    integral = 0
     
    for quad_point_x in quad_points_x:
        for quad_point_y in quad_points_y:
            integral += f(quad_point_x.point[0], quad_point_x.point[1], quad_point_y.point[0], quad_point_y.point[1]) * basis_function_i.function(quad_point_x.point) * basis_function_j.function(quad_point_y.point) * quad_point_x.weight * quad_point_y.weight
    return integral

def calculate_vector_field_eigenpairs(mesh_resolution):  
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
        vertex_coords = np.array([mesh.coordinates()[vertex] for vertex in cell.entities(0)])

        basis_functions.append(ConstBasisFunction(basis_function, vertex_coords))

    C = np.zeros((2 * N, 2 * N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if j <= i:
                # Here we use that each block is symmetric because of the symmetry of the covariance functions
                C[i, j] = C[j, i] = get_C_entry(v_cov1_1, basis_function_i, basis_function_j)
                C[i, N + j] = C[j, N + i] = get_C_entry(v_cov1_2, basis_function_i, basis_function_j)
                C[N + i, j] = C[N + j, i] = get_C_entry(v_cov2_1, basis_function_i, basis_function_j)
                C[N + i, N + j] = C[N + j, N + i] = get_C_entry(v_cov2_2, basis_function_i, basis_function_j)

    M = np.zeros((2 * N, 2 * N))
    for i, basis_function_i in enumerate(basis_functions):
        integrand = basis_function_i.function * basis_function_i.function * fe.dx
        M[i, i] = M[N + i, N + i] = fe.assemble(integrand)

    J = N # Number of eigenvectors -> J = N is maximum
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
    return RandomFieldV(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J), \
            JacobianV(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J)
    
def solve_poisson_for_given_sample(fem_mesh_resolution, jacobianV, xi, f):
    jacobianV_fixed_xi = JacobianVFixedXi(jacobianV, xi)
    mesh = mshr.generate_mesh(DOMAIN, fem_mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 1)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    A_expr = AExpression(jacobianV_fixed_xi, degree=2)
    a = fe.inner(fe.dot(A_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV_fixed_xi, degree=2)
    L = f * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    u_sol.set_allow_extrapolation(True)
    return u_sol

def save_results_to_csv(u_sols_evaluated: np.array, xis: np.array, fem_res: int, kl_res: int):
    i = 0
    samples_filename = f'poisson_sample_storage/samples_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    # xis_filename = f'poisson_sample_storage/xis_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    
    # Increment i until a non-existing filename is found
    while os.path.exists(samples_filename):
        i += 1
        samples_filename = f'poisson_sample_storage/samples_femres_{fem_res}_klres_{kl_res}_{i}.csv'
        # xis_filename = f'poisson_sample_storage/xis_femres_{fem_res}_klres_{kl_res}_{i}.csv'
    
    np.savetxt(samples_filename, u_sols_evaluated, delimiter=',', fmt="%.18e")
    # np.savetxt(xis_filename, xis, delimiter=',', fmt="%.18e")
    print(f"u_sols_evaluated saved to {samples_filename}")

def calculate_samples_and_save_results(mc_samples: int, fem_res: int, kl_res: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None):

    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    # Take fem_res for saving scheme
    mesh = mshr.generate_mesh(DOMAIN, fem_res)
    V = fe.FunctionSpace(mesh, "CG", 1)
    dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

    u_sols = []
    u_sols_evaluated = np.zeros((mc_samples, len(dof_coordinates)))
    xis = np.zeros((mc_samples, randomFieldV.J))
    for i in range(mc_samples):
        print(f"Iteration {i+1}/{mc_samples}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), randomFieldV.J)
        xis[i] = xi
        u_sols.append(solve_poisson_for_given_sample(fem_res, jacobianV, xi, RHS_F))
        for j, point_coords in enumerate(dof_coordinates):
            u_sols_evaluated[i, j] = u_sols[i](fe.Point(point_coords))
    save_results_to_csv(u_sols_evaluated, xis, fem_res, kl_res)

def analyse_two_resolutions_from_data_u_hat(resolution_sparse, resolution_fine, P_hat):
    # Here the sparse resolution is used for both fem_res and the fine resolution is used just for the kl_res of the fine solution
    # So for both functionspaces the sparse mesh resolution is used for the mesh
    data_sparse = np.genfromtxt(f'poisson_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_sparse}_combined.csv', delimiter=',')
    data_fine = np.genfromtxt(f'poisson_sample_storage/samples_femres_{resolution_sparse}_klres_{resolution_fine}_combined.csv', delimiter=',')

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

    print("Data dimensions: sparse: ", data_sparse.shape, "fine: ", data_fine.shape)
    print("MC Sample sizes sparse: ", mc_sample_sizes)


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
    fig_mean = go.Figure(data=[go.Surface(z=grid_z_mean, x=grid_x, y=grid_y, colorscale='Viridis')])
    fig_mean.update_layout(title=dict(text="Mean estimation û(x,y)", x=0.5, y=0.95),
                    autosize=True,
                    # height=400,
                    margin=dict(l=10, r=10, b=10, t=20),
                    scene=dict(
                        xaxis_title='x-axis',
                        yaxis_title='y-axis',
                        zaxis_title='û(x,y)'))
    fig_mean.show()

    grid_z_var = griddata((x_coords, y_coords), z_values_fine_var, (grid_x, grid_y), method='linear')
    fig_var = go.Figure(data=[go.Surface(z=grid_z_var, x=grid_x, y=grid_y, colorscale='Viridis', colorbar=dict(exponentformat='e'))])
    fig_var.update_layout(title=dict(text="Variance estimation û(x,y)", x=0.5, y=0.95),
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

    plt.suptitle(f'Means and Variance of û(x,y) in point ({P_hat.x()}, {P_hat.y()})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot L2 and H1 errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(mc_sample_sizes, L2_errors, 'bo', marker='x', label='L2 Error')
    ax1.set_xscale('log')
    #ax1.set_yscale('log')
    ax1.set_xlabel('MC Samples')
    ax1.set_ylabel('L2 Error')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(mc_sample_sizes, H1_errors, 'bo', marker='x', label='H1 Error')
    ax2.set_xscale('log')
    #ax2.set_yscale('log')
    ax2.set_xlabel('MC Samples')
    ax2.set_ylabel('H1 Error')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.suptitle('L2 and H1 Errors of û(x,y) to reference solution')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def measure_time_one_sample(len_xi, kl_res, fem_res, randomFieldV, jacobianV):

    xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), len_xi)

    if len_xi > randomFieldV.J:
        raise ValueError("len(xi) must be less than randomFieldV.J")

    time_start = time.time()
    u_sol = solve_poisson_for_given_sample(fem_res, jacobianV, xi, RHS_F)
    time_end = time.time()
    print(f"len_xi: {len_xi}, fem_res: {fem_res}, kl_res: {kl_res}")
    print(f"Time: {time_end - time_start} seconds")
    print("____")

def plot_mesh(mesh_resolution, randomFieldV):
    xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), randomFieldV.J)
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    # perturbed mesh based on the "original" mesh used for the KL-expansion
    perturbed_coordinates = mesh.coordinates().copy()
    for index, coordinate in enumerate(mesh.coordinates()):
        perturbed_coordinates[index] = randomFieldV(coordinate, xi)
    # Create a new mesh with the perturbed coordinates
    perturbed_mesh = fe.Mesh(mesh)
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    # Plot the original mesh and the perturbed mesh in one figure with different colors
    plt.figure()
    fe.plot(mesh, color='blue', linewidth=0.5, label='Original Mesh')
    fe.plot(perturbed_mesh, color='red', linewidth=0.5, label='Perturbed Mesh')
    plt.legend()
    plt.title(f"Mesh, mesh_resolution: {mesh_resolution}, N: {randomFieldV.N}")
    plt.show()

def incoherence_fe_edges_visualisation(randomFieldV, mesh_resolution):

    xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), randomFieldV.J)

    number_of_points_per_edge = 100

    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    orig_tri = next(fe.cells(mesh))
    midpoint = orig_tri.midpoint().array()[0:2]
    orig_tri_coords = np.array(orig_tri.get_vertex_coordinates()).reshape((-1, 2))
    print(f"orig_tri_coords:{orig_tri_coords}")
    orig_points = []
    orig_points_small = []
    for i in range(number_of_points_per_edge):
        orig_points.append(i/number_of_points_per_edge * orig_tri_coords[0] + (1 - i/number_of_points_per_edge) * orig_tri_coords[1])
        orig_points.append(i/number_of_points_per_edge * orig_tri_coords[1] + (1 - i/number_of_points_per_edge) * orig_tri_coords[2])
        orig_points.append(i/number_of_points_per_edge * orig_tri_coords[2] + (1 - i/number_of_points_per_edge) * orig_tri_coords[0])
        orig_points_small.append(1/2 *(i/number_of_points_per_edge * orig_tri_coords[0] + (1 - i/number_of_points_per_edge) * orig_tri_coords[1]) + 1/2 * midpoint)
        orig_points_small.append(1/2 * (i/number_of_points_per_edge * orig_tri_coords[1] + (1 - i/number_of_points_per_edge) * orig_tri_coords[2]) + 1/2 * midpoint)
        orig_points_small.append(1/2 * (i/number_of_points_per_edge * orig_tri_coords[2] + (1 - i/number_of_points_per_edge) * orig_tri_coords[0]) + 1/2 * midpoint)

    mapped_points = []
    for orig_point in orig_points:
        mapped_points.append(randomFieldV(orig_point, xi))
    mapped_points_small = []
    for orig_point_small in orig_points_small:
        mapped_points_small.append(randomFieldV(orig_point_small, xi))

    orig_points = np.array(orig_points)
    mapped_points = np.array(mapped_points)
    orig_points_small = np.array(orig_points_small)
    mapped_points_small = np.array(mapped_points_small)

    plt.figure()

    plt.scatter(orig_points[:, 0], orig_points[:, 1], color='green', label='Original Points', marker='.')
    plt.scatter(mapped_points[:, 0], mapped_points[:, 1], color='red', label='Mapped Points', marker='.')

    plt.scatter(orig_points_small[:, 0], orig_points_small[:, 1], color='blue', label='Original Points - smaller triangle', marker='.')
    plt.scatter(mapped_points_small[:, 0], mapped_points_small[:, 1], color='cyan', label='Mapped Points - smaller triangle', marker='.')

    plt.legend()
    plt.title('Original and Mapped Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def inverse_mapping(P, randomFieldV, xi, mesh_resolution_inverse_mapping):
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution_inverse_mapping)
    perturbed_coordinates = mesh.coordinates().copy()
    for index, coordinate in enumerate(mesh.coordinates()):
        perturbed_coordinates[index] = randomFieldV(coordinate, xi)
    perturbed_mesh = fe.Mesh(mesh)
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    image_tria_coords = None
    for cell in fe.cells(perturbed_mesh):
        if cell.contains(P):
            image_tria_coords = np.array(cell.get_vertex_coordinates()).reshape((-1, 2))
            break

    if image_tria_coords is None:
        raise ValueError("Point P is not inside any cell of the perturbed mesh")

    image_bary_coords = barycentric_coords(P, image_tria_coords)

    perturbed_coords = perturbed_mesh.coordinates()
    original_coords = mesh.coordinates()

    indices = []
    for vertex in image_tria_coords:
        for i, coord in enumerate(perturbed_coords):
            if np.allclose(vertex, coord):
                indices.append(i)
                break

    original_tria_coords = original_coords[indices]

    P_hat = (
        image_bary_coords[0] * original_tria_coords[0] +
        image_bary_coords[1] * original_tria_coords[1] +
        image_bary_coords[2] * original_tria_coords[2]
    )

    return P_hat

def barycentric_coords(P, vertices):
    A = np.array([
        [vertices[0][0], vertices[1][0], vertices[2][0]],
        [vertices[0][1], vertices[1][1], vertices[2][1]],
        [1, 1, 1]
    ])
    b = np.array([P.x(), P.y(), 1])
    bary_coords = np.linalg.solve(A, b)
    return bary_coords

# Sobol

def poisson_sobol_run_samples_and_save(mc_sample_size: int, fem_res: int, kl_res: int, size_of_xi: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None) -> None:

    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)
    
    # Collect points P
    N = len(randomFieldV.basis_functions)
    
    # Sample all xis, not only the ones that are investigated
    len_xi_total = randomFieldV.J
    A = np.zeros((mc_sample_size, len_xi_total))
    B = np.zeros((mc_sample_size, len_xi_total))

    for xi_index in range(len_xi_total):
        A[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)
        B[:, xi_index] = np.random.uniform(-np.sqrt(3), np.sqrt(3), mc_sample_size)

    f_A = np.zeros((mc_sample_size, N))
    f_B = np.zeros((mc_sample_size, N))

    for m in range(mc_sample_size):
        print(f"First loop Iteration {m+1}/{mc_sample_size}")
        u_sol_A = solve_poisson_for_given_sample(fem_res, jacobianV, A[m], RHS_F)
        u_sol_B = solve_poisson_for_given_sample(fem_res, jacobianV, B[m], RHS_F)
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

    for i in range(size_of_xi):
        C_i = np.zeros((mc_sample_size, len_xi_total))
        for param_index in range(len_xi_total):
            if param_index == i:
                C_i[:, param_index] = A[:, param_index]
            else:
                C_i[:, param_index] = B[:, param_index]
        
        f_C_i = np.zeros((mc_sample_size, N))
        for m in range(mc_sample_size):
            print(f"Second loop xi {i+1} of {size_of_xi}, Iteration {m+1}/{mc_sample_size}")
            u_sol_C_i = solve_poisson_for_given_sample(fem_res, jacobianV, C_i[m], RHS_F)
            for basis_function_index, basis_function in enumerate(randomFieldV.basis_functions):
                transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
                active_quad_points = QUAD_POINTS_2DD_6
                quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
                int_C_i = 0
                for quad_point in quad_points:
                    int_C_i += u_sol_C_i(quad_point.point) * quad_point.weight
                f_C_i[m, basis_function_index] = int_C_i
        f_C_is.append(f_C_i)
        
    base_path = f'sobol_data_storage/poisson/femres_{fem_res}_klres_{kl_res}_size_of_xi_{size_of_xi}'
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

    for i in range(size_of_xi):
        f_C_i_path = os.path.join(base_path, f'f_C_{i}.npy')
        if os.path.exists(f_C_i_path):
            f_C_i_existing = np.load(f_C_i_path)
            f_C_is[i] = np.concatenate((f_C_i_existing, f_C_is[i]), axis=0)
        np.save(f_C_i_path, f_C_is[i])

    print("Sobol data saved")

def poisson_sobol_calc_indices_from_data(fem_res: int, kl_res: int, size_of_xi: int, randomFieldV: RandomFieldV = None, jacobianV: JacobianV = None) -> tuple[np.array, np.array, int]:

    if not randomFieldV or not jacobianV:
        randomFieldV, jacobianV = calculate_vector_field_eigenpairs(kl_res)

    base_path = f'sobol_data_storage/poisson/femres_{fem_res}_klres_{kl_res}_size_of_xi_{size_of_xi}'
    f_A_path = os.path.join(base_path, 'f_A.npy')
    f_B_path = os.path.join(base_path, 'f_B.npy')

    f_A = np.load(f_A_path)
    f_B = np.load(f_B_path)

    S_single = np.zeros(size_of_xi)
    S_total = np.zeros(size_of_xi)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)
    weights = [basis_function.triangle_area for basis_function in randomFieldV.basis_functions]

    for i in range(size_of_xi):
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

def poisson_plot_sobols(S_single, S_total, mc_sample_size, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set width for each bar
    bar_width = 0.35

    ax.bar(np.arange(len(S_single)), S_single, width=bar_width, label='First Order')
    ax.bar(np.arange(len(S_single)) + bar_width, S_total, width=bar_width, label='Total Effect')
    x_labels = [fr"$\xi_{{{i+1}}}$" for i in range(len(S_single))]
    ax.set_xticklabels(x_labels)
    ax.set_xticks(np.arange(len(S_single)) + bar_width / 2)
    ax.set_ylabel('Sensitivity [-]')
    ax.set_title(f'{title} Sample Size: {mc_sample_size}')
    ax.grid(True)
    ax.legend()
    plt.show()

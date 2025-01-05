import matplotlib.pyplot as plt
import fenics as fe
import mshr
import numpy as np
from scipy.linalg import eigh

# Constants
CENTER = fe.Point(0, 0)
RADIUS = 1
DOMAIN = mshr.Circle(CENTER, RADIUS)
RHS_F = fe.Constant(1)
DIRICHLET_BC = fe.Constant(0)

# Covariance functions
#TODO als Formeln abspeichern damit man Ableitung bilden kann oder selbst ausrechnen
def v_cov1_1(x, y):
    return 5.0/100.0 * np.exp(-4.0 * ((x[0] - y[0])**2 + (x[1] - y[1])**2))
def v_cov1_2(x, y):
    return 1.0/100.0 * np.exp(-0.1 * ((2*x[0] - y[0])**2 + (2*x[1] - y[1])**2))
def v_cov2_1(x, y):
    return 1.0/100.0 * np.exp(-0.1 * ((x[0] - 2*y[0])**2 + (x[1] - 2*y[1])**2))
def v_cov2_2(x, y):
    return 5.0/100.0 * np.exp(-1.0 * ((x[0] - y[0])**2 + (x[1] - y[1])**2))

class V_cov1_1_Expression(fe.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 5.0/100.0 * np.exp(-4.0 * ((x[0] - y[0])**2 + (x[1] - y[1])**2))

    def value_shape(self):
        return (2, 2)



# Classes
class BasisFunction():
    def __init__(self, basis_function: fe.Function, coordinates: np.array):
        self.function = basis_function
        self.coordinates = coordinates

class RandomFieldV():
    def __init__(self, eigenvalues, eigenvectors, basis_functions, N, J):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J

    def __call__(self, x, xi):
        return x[0] + sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[k, j] * self.basis_functions[k].function(x) for k in range(self.N)]) * xi[j] for j in range(len(xi))]), \
           x[1] + sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[self.N + k, j] * self.basis_functions[k].function(x) for k in range(self.N)]) * xi[j] for j in range(len(xi))])

class JacobianV():
    def __init__(self, eigenvalues, eigenvectors, basis_functions_grads, N: int, J: int):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions_grads = basis_functions_grads
        self.N = N
        self.J = J

    def __call__(self, x, xi):
        #TODO Hier andere Funktion nehmen aus Ullmann Paper
        jacobian_output = np.zeros((2, 2))
        jacobian_output[0, 0] = 1 + sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[k, j] * self.basis_functions_grads[k].function(x)[0] for k in range(self.N)]) * xi[j] for j in range(len(xi))])
        jacobian_output[0, 1] = sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[self.N + k, j] * self.basis_functions_grads[k].function(x)[0] for k in range(self.N)]) * xi[j] for j in range(len(xi))])
        jacobian_output[1, 0] = sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[k, j] * self.basis_functions_grads[k].function(x)[1] for k in range(self.N)]) * xi[j] for j in range(len(xi))])
        jacobian_output[1, 1] = 1 + sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[self.N + k, j] * self.basis_functions_grads[k].function(x)[0] for k in range(self.N)]) * xi[j] for j in range(len(xi))])
        return jacobian_output
    

class AExpression(fe.UserExpression):
    def __init__(self, jacobianV, xi, **kwargs):
        super().__init__(**kwargs)
        self.jacobianV = jacobianV
        self.xi = xi

    def eval(self, values, x):
        J_x = self.jacobianV(x, self.xi)
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
    def __init__(self, jacobianV, xi, **kwargs):
        super().__init__(**kwargs)
        self.jacobianV = jacobianV
        self.xi = xi

    def eval(self, values, x):
        J_x = self.jacobianV(x, self.xi)
        det_J = np.linalg.det(J_x)
        values[0] = det_J

    def value_shape(self):
        return ()

class RandomFieldExpression(fe.UserExpression):
    def __init__(self, randomfieldV, xi, **kwargs):
        super().__init__(**kwargs)
        self.randomfieldV = randomfieldV
        self.xi = xi

    def eval(self, values, x):
        values[0] = self.randomfieldV(x, self.xi)
    
    def value_shape(self):
        return ()


# Methods

def are_points_same(p1, p2):
    return np.allclose(p1, p2)

def are_points_neighbors(mesh, p1, p2):
    vertex_to_point = {tuple(vertex.point().array()[:2]): vertex.index() for vertex in fe.vertices(mesh)}
    if tuple(p1) not in vertex_to_point or tuple(p2) not in vertex_to_point:
        return False
    v1 = vertex_to_point[tuple(p1)]
    v2 = vertex_to_point[tuple(p2)]    
    for cell in fe.cells(mesh):
        vertices = cell.entities(0)
        if v1 in vertices and v2 in vertices:
            return True
    return False

def get_shared_triangles(mesh, p1, p2):
    #! This method is not efficient and should be optimized
    #! Use Hashmap for neighbors instead
    vertex_to_point = {tuple(vertex.point().array()[:2]): vertex.index() for vertex in fe.vertices(mesh)}
    v1 = vertex_to_point[tuple(p1)]
    v2 = vertex_to_point[tuple(p2)]
    shared_triangles = []
    for cell in fe.cells(mesh):
        vertices = cell.entities(0)
        if v1 in vertices and v2 in vertices:
            vertex_coords = np.array([cell.mesh().coordinates()[vertex][:2] for vertex in vertices])
            shared_triangles.append(vertex_coords)
    return shared_triangles

def find_affine_transformation(triangle):
    transformation_matrix = np.array([[triangle[1, 0] - triangle[0, 0], triangle[2, 0] - triangle[0, 0]],
                                    [triangle[1, 1] - triangle[0, 1], triangle[2, 1] - triangle[0, 1]]])
    transformation_vector = np.array([triangle[0, 0], triangle[0, 1]])
    return transformation_matrix, transformation_vector

class quad_point():
    def __init__(self, point: list, weight: float):
        self.point = point
        self.weight = weight

def triangle_area(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def integrate_over_triangles(f, triangle_x, triangle_y, basis_function_i, basis_function_j):
    transformation_matrix_x, transformation_vector_x = find_affine_transformation(triangle_x)
    transformation_matrix_y, transformation_vector_y = find_affine_transformation(triangle_y)
    quad_points_2DD_5 = [quad_point([0, 0], 3/120),
                         quad_point([1, 0], 3/120),
                         quad_point([0, 1], 3/120),
                         quad_point([1/2, 0], 8/120),
                         quad_point([1/2, 1/2], 8/120),
                         quad_point([0, 1/2], 8/120),
                         quad_point([1/3, 1/3], 27/120)]
    quad_points_2DD_6 = [quad_point([(6 - np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                         quad_point([(9 + 2 * np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                         quad_point([(6 - np.sqrt(15)) / 21, (9 + 2 * np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                         quad_point([(6 + np.sqrt(15)) / 21, (9 - 2 * np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                         quad_point([(6 + np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                         quad_point([(9 - 2 * np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                         quad_point([1 / 3, 1 / 3], 9/80)]
    integral = 0
    for quad_point_x in quad_points_2DD_6:
        quad_point_x.point = np.dot(transformation_matrix_x, quad_point_x.point) + transformation_vector_x
        for quad_point_y in quad_points_2DD_6:
            quad_point_y.point = np.dot(transformation_matrix_y, quad_point_y.point) + transformation_vector_y
            integral += f(quad_point_x.point, quad_point_y.point) * basis_function_i.function(quad_point_x.point) * basis_function_j.function(quad_point_y.point) * quad_point_x.weight * quad_point_y.weight
    area_x = triangle_area(triangle_x)
    area_y = triangle_area(triangle_y)
    return integral * 2 * area_x * 2 * area_y # weights for unit triangle so this is the factor for our triangles

def get_C_entry(mesh, f, basis_function_i: BasisFunction, basis_function_j: BasisFunction):
    integral = 0
    if are_points_same(basis_function_i.coordinates, basis_function_j.coordinates):
        # case basis_functions are the same
        shared_triangles = get_shared_triangles(mesh, basis_function_i.coordinates, basis_function_j.coordinates)
        for triangle_x in shared_triangles:
            for triangle_y in shared_triangles:
                integral += integrate_over_triangles(f, triangle_x, triangle_y, basis_function_i, basis_function_j)
    elif(are_points_neighbors(mesh, basis_function_i.coordinates, basis_function_j.coordinates)):       
        # case basis_functions are neighbors
        shared_triangles = get_shared_triangles(mesh, basis_function_i.coordinates, basis_function_j.coordinates)
        for triangle_x in shared_triangles:
            for triangle_y in shared_triangles:
                integral += integrate_over_triangles(f, triangle_x, triangle_y, basis_function_i, basis_function_j)
    return integral

def calculate_vector_field_eigenpairs(mesh_resolution_c_entries):  
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution_c_entries)
    V = fe.FunctionSpace(mesh, "CG", 1)
    V_Vector = fe.VectorFunctionSpace(mesh, "CG", 1)
    N = V.dim()
    dof_coordinates = V.tabulate_dof_coordinates().reshape((-1, 2))

    basis_functions = []
    basis_functions_grads = []
    for i in range(N):
        basis_function = fe.Function(V)
        basis_function.vector()[i] = 1.0
        basis_function.set_allow_extrapolation(True)
        basis_functions.append(BasisFunction(basis_function, dof_coordinates[i]))
        grad = fe.project(fe.grad(basis_function), V_Vector)
        grad.set_allow_extrapolation(True)
        basis_functions_grads.append(BasisFunction(grad, dof_coordinates[i]))

    C = np.zeros((2 * N, 2 * N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if j <= i:
                # Here we use that each block is symmetric because of the symmetry of the covariance functions
                C[i, j] = C[j, i] = get_C_entry(mesh, v_cov1_1, basis_function_i, basis_function_j)
                C[i, N + j] = C[j, N + i] = get_C_entry(mesh, v_cov1_2, basis_function_i, basis_function_j)
                C[N + i, j] = C[N + j, i] = get_C_entry(mesh, v_cov2_1, basis_function_i, basis_function_j)
                C[N + i, N + j] = C[N + j, N + i] = get_C_entry(mesh, v_cov2_2, basis_function_i, basis_function_j)
    print(f"C: {C}")

    M = np.zeros((2 * N, 2 * N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if i >= j:
                integrand = basis_function_j.function * basis_function_i.function * fe.dx
                M[i, j] = M[j, i] = M[N + i, N + j] = M[N + j, N + i] = fe.assemble(integrand)

    J = N # Number of eigenvectors -> J = N is maximum
    eigenvalues, eigenvectors = eigh(C, M, subset_by_index=[0, J-1])
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Eliminate negative eigenvalues
    for index, sorted_eigenvalue in enumerate(sorted_eigenvalues):
        if sorted_eigenvalue < 0:
            sorted_eigenvalues[index] = 0
    return RandomFieldV(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J), \
            JacobianV(sorted_eigenvalues, sorted_eigenvectors, basis_functions_grads, N, J)
    
def solve_poisson_for_given_sample(mesh_resolution, jacobianV, xi, f):
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 3)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    A_expr = AExpression(jacobianV, xi, degree=2)
    a = fe.inner(fe.dot(A_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV, xi, degree=2)
    L = f * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    return u_sol

def mc_single_global_calculation(mc_samples_single: int, mesh_resolution_fem_single: int, jacobianV: JacobianV):
    u_sols = []
    for i in range(mc_samples_single):
        print(f"Iteration {i+1}/{mc_samples_single}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), jacobianV.J)
        u_sols.append(solve_poisson_for_given_sample(mesh_resolution_fem_single, jacobianV, xi, RHS_F))

    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution_fem_single)
    V = fe.FunctionSpace(mesh, "CG", 3)
    mean_sol = fe.Function(V)
    mean_sol.vector()[:] = 0
    for u_sol in u_sols:
        mean_sol.vector()[:] += u_sol.vector() / mc_samples_single

    #! Not sure about the variance calculation in the degrees of freedom
    var_sol = fe.Function(V)
    var_sol.vector()[:] = 0
    for u_sol in u_sols:
        for dof_index in range(len(u_sol.vector())):
            if mc_samples_single > 1:
                var_sol.vector()[dof_index] += (u_sol.vector()[dof_index] - mean_sol.vector()[dof_index])**2 / (mc_samples_single-1)
    return mean_sol, var_sol


# Helpers for multiple mc loops local analysis

def true_sol(mesh_resolution_fem_true_sol):
    #! TRUE SOLUTION OF UN-PERTURBED! I don't know if rhs and u_hat != u for true solution
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution_fem_true_sol)
    V = fe.FunctionSpace(mesh, "CG", 3) # Consider higher order if mc simulations are more accurate
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    ref_a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    L = RHS_F * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_true_sol = fe.Function(V)
    fe.solve(ref_a == L, u_true_sol, bc)    
    return u_true_sol

def mc_multiple_local_calculation(mc_samples_multiple, mesh_resolution_fem_multiple, mesh_resolution_fem_true_sol, P, jacobianV):
    u_sols_in_point_P_mean = np.zeros(len(mc_samples_multiple))
    u_sols_in_point_P_var = np.zeros(len(mc_samples_multiple))
    L_2_errors = np.zeros(len(mc_samples_multiple))
    H_1_errors = np.zeros(len(mc_samples_multiple))
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution_fem_multiple)
    V = fe.FunctionSpace(mesh, "CG", 3)
    u_true_sol = true_sol(mesh_resolution_fem_true_sol)

    for mc_sample_size_index, mc_sample_size in enumerate(mc_samples_multiple):
        u_sols_in_point_P = []
        mean_sol = fe.Function(V)
        mean_sol.vector()[:] = 0
        for i in range(mc_sample_size):
            print(f"Iteration {i+1}/{mc_sample_size}")
            xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), jacobianV.J)
            u_sol = solve_poisson_for_given_sample(mesh_resolution_fem_multiple, jacobianV, xi, RHS_F)
            u_sols_in_point_P.append(u_sol(P))
            mean_sol.vector()[:] += u_sol.vector() / mc_sample_size

        #! Plot of means
        c = fe.plot(mean_sol, title='Solution with MC-sample size ' + str(mc_sample_size))
        plt.colorbar(c)
        plt.show()

        u_sols_in_point_P_mean[mc_sample_size_index] = np.mean(u_sols_in_point_P)
        u_sols_in_point_P_var[mc_sample_size_index] = np.var(u_sols_in_point_P, ddof=1)
        L_2_errors[mc_sample_size_index] = fe.errornorm(u_true_sol, mean_sol, norm_type='L2')
        H_1_errors[mc_sample_size_index] = fe.errornorm(u_true_sol, mean_sol, norm_type='H1')
    return u_sols_in_point_P_mean, u_sols_in_point_P_var, L_2_errors, H_1_errors, u_true_sol(P)




# Helpers Sensitivity Analysis

def double_loop_mc(mc_sample_size, mesh_resolution_fem, P, indices, jacobianV, size_of_xi):
    # Translate math indices to CS indices
    indices = [index - 1 for index in indices]

    # Implementation of the double loop MC estimation of specific Sobol Index
    mean_sols_point_P = np.zeros(mc_sample_size)
    for i in range(mc_sample_size):
        print(f"First loop: Outer MC loop iteration {i+1} / {mc_sample_size}")
        sols_point_P = []
        sample = np.zeros(size_of_xi + 1)
        for index in indices:
            if index < size_of_xi:
                sample[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
            else:
                sample[index] = np.random.normal(0, 1)
        missing_indices = [i for i in range(size_of_xi + 1) if i not in indices]
        for k in range(mc_sample_size):
            for index in missing_indices:
                if index < size_of_xi:
                    sample[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                else:
                    sample[index] = np.random.normal(0, 1)
            sols_point_P.append(solve_poisson_for_given_sample(mesh_resolution_fem, jacobianV, sample[0:-1], sample[-1])(P))
        mean_sols_point_P[i] = np.mean(sols_point_P)
    var_A = np.var(mean_sols_point_P, ddof=1)
    sols_point_P = []
    for i in range(mc_sample_size):
        print(f"Second loop: {i+1} / {mc_sample_size}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        f = np.random.normal(0, 1)
        sols_point_P.append(solve_poisson_for_given_sample(mesh_resolution_fem, jacobianV, xi, f)(P))
    var = np.var(sols_point_P, ddof=1)
    return var_A / var


def pick_freeze(mc_sample_size, mesh_resolution_fem, P, indices, jacobianV, size_of_xi):
    # Translate math indices to CS indices
    indices = [index - 1 for index in indices]

    y = []
    y_tilde = []
    for i in range(mc_sample_size):
        print(f"First loop: {i+1} / {mc_sample_size}")
        x = np.zeros(size_of_xi + 1)
        x_prime = np.zeros(size_of_xi + 1)
        for index in indices:
            if index < size_of_xi:
                sample = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                x[index] = sample
                x_prime[index] = sample
            else:
                sample = np.random.normal(0, 1)
                x[index] = sample
                x_prime[index] = sample
        for index in [i for i in range(size_of_xi + 1) if i not in indices]:
            if index < size_of_xi:
                x[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                x_prime[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
            else:
                x[index] = np.random.normal(0, 1)
                x_prime[index] = np.random.normal(0, 1)
        y.append(solve_poisson_for_given_sample(mesh_resolution_fem, jacobianV, x[0:-1], x[-1])(P))
        y_tilde.append(solve_poisson_for_given_sample(mesh_resolution_fem, jacobianV, x_prime[0:-1], x_prime[-1])(P))
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
        f = np.random.normal(0, 1)
        sols_point_P.append(solve_poisson_for_given_sample(mesh_resolution_fem, jacobianV, xi, f)(P))
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
    return sorted_helper[:, 0]
    
def rank_statistics(mc_sample_size, mesh_resolution_fem, P, index, jacobianV, size_of_xi):
    # Translate math index to CS index
    index -= 1

    y = []
    samples = np.zeros((mc_sample_size, size_of_xi + 1))
    for i in range(mc_sample_size):
        print(f"Loop: {i+1} / {mc_sample_size}")
        samples[i, 0:-1] = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        samples[i, -1] = np.random.normal(0, 1)
        y.append(solve_poisson_for_given_sample(mesh_resolution_fem, jacobianV, samples[i, 0:-1], samples[i, -1])(P))
    y = rank_statistics_permute(samples[:, index].tolist(), y)
    y_bar = np.mean(y)
    V_hat_j = 1/mc_sample_size * sum([y[i] * y[P_j(i, sorted(samples[:, index].tolist()), samples[:, index].tolist())] for i in range(mc_sample_size)]) - y_bar**2
    V_hat = 1/mc_sample_size * sum([y[i]**2 for i in range(mc_sample_size)]) - y_bar**2
    return V_hat_j / V_hat
import fenics as fe
import mshr
import numpy as np
from scipy.linalg import eig

# Constants
CENTER = fe.Point(0, 0)
RADIUS = 1
DOMAIN = mshr.Circle(CENTER, RADIUS)
RHS_F = fe.Constant(1)
DIRICHLET_BC = fe.Constant(0)

def find_affine_transformation(triangle):
    transformation_matrix = np.array([[triangle[1, 0] - triangle[0, 0], triangle[2, 0] - triangle[0, 0]],
                                    [triangle[1, 1] - triangle[0, 1], triangle[2, 1] - triangle[0, 1]]])
    transformation_vector = np.array([triangle[0, 0], triangle[0, 1]])
    return transformation_matrix, transformation_vector

class quad_point():
    def __init__(self, point: list, weight: float):
        self.point = point
        self.weight = weight

QUAD_POINTS_2DD_5 = [quad_point([0, 0], 3/120),
                        quad_point([1, 0], 3/120),
                        quad_point([0, 1], 3/120),
                        quad_point([1/2, 0], 8/120),
                        quad_point([1/2, 1/2], 8/120),
                        quad_point([0, 1/2], 8/120),
                        quad_point([1/3, 1/3], 27/120)]
QUAD_POINTS_2DD_6 = [quad_point([(6 - np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        quad_point([(9 + 2 * np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        quad_point([(6 - np.sqrt(15)) / 21, (9 + 2 * np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        quad_point([(6 + np.sqrt(15)) / 21, (9 - 2 * np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        quad_point([(6 + np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        quad_point([(9 - 2 * np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        quad_point([1 / 3, 1 / 3], 9/80)]

def triangle_area(vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    # which is the determinant of the transformation matrix
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))


class BasisFunction():
    def __init__(self, basis_function: fe.Function, supp_cells: list):
        self.function = basis_function
        self.supp_cells = supp_cells

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


def get_C_entry(f, basis_function_i: BasisFunction, basis_function_j: BasisFunction):

    active_quad_points = QUAD_POINTS_2DD_6

    integral = 0

    for cell_x in basis_function_i.supp_cells:
        transformation_matrix_x, transformation_vector_x = find_affine_transformation(np.array(cell_x.get_vertex_coordinates()).reshape((-1, 2)))
        area_x = triangle_area(np.array(cell_x.get_vertex_coordinates()).reshape((-1, 2)))
        quad_points_x = [quad_point(np.dot(transformation_matrix_x, quad_point_x.point) + transformation_vector_x, quad_point_x.weight * 2 * area_x) for quad_point_x in active_quad_points]

        for cell_y in basis_function_j.supp_cells:
            transformation_matrix_y, transformation_vector_y = find_affine_transformation(np.array(cell_y.get_vertex_coordinates()).reshape((-1, 2)))
            area_y = triangle_area(np.array(cell_y.get_vertex_coordinates()).reshape((-1, 2)))
            quad_points_y = [quad_point(np.dot(transformation_matrix_y, quad_point_y.point) + transformation_vector_y, quad_point_y.weight * 2 * area_y) for quad_point_y in active_quad_points]

            for quad_point_x in quad_points_x:
                for quad_point_y in quad_points_y:
                    integral += f(quad_point_x.point, quad_point_y.point) * basis_function_i.function(quad_point_x.point) * basis_function_j.function(quad_point_y.point) * quad_point_x.weight * quad_point_y.weight
    return integral

def calculate_vector_field_eigenpairs(mesh_resolution, v_cov1_1, v_cov1_2, v_cov2_1, v_cov2_2):  
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "P", 3)
    V_Vector = fe.VectorFunctionSpace(mesh, "P", 3)
    
    basis_functions = []
    basis_functions_grads = []

    for i in range(V.dim()):
        basis_function = fe.Function(V)
        basis_function.vector()[i] = 1
        for vertex in fe.vertices(mesh):
            if abs(basis_function(vertex.point().array()[:2]) - 1) < 1e-6:
                basis_function.set_allow_extrapolation(True)
                grad = fe.project(fe.grad(basis_function), V_Vector)
                grad.set_allow_extrapolation(True)
                supp_cells = [cell for cell in fe.cells(mesh) if basis_function(cell.midpoint().array()[:2]) != 0]
                basis_functions.append(BasisFunction(basis_function, supp_cells))
                basis_functions_grads.append(BasisFunction(grad, supp_cells))
    
    N = len(basis_functions)

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
        for j, basis_function_j in enumerate(basis_functions):
            integrand = basis_function_i.function * basis_function_j.function * fe.dx
            M[i, j] = M[N + i, N + j] = fe.assemble(integrand)

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

def barycentric_coords(P, vertices):
    A = np.array([
        [vertices[0][0], vertices[1][0], vertices[2][0]],
        [vertices[0][1], vertices[1][1], vertices[2][1]],
        [1, 1, 1]
    ])
    b = np.array([P.x(), P.y(), 1])
    bary_coords = np.linalg.solve(A, b)
    return bary_coords


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


def true_sol(mesh_resolution_fem_true_sol):
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
    


def non_varying_area(len_xi, randomFieldV):   
    R_nv = np.sqrt(3) * np.sqrt((np.sum([np.sqrt(randomFieldV.eigenvalues[m]) * 3 * np.max([np.abs(randomFieldV.eigenvectors[j, m]) for j in range(randomFieldV.N)]) for m in range(len_xi)]))**2 \
                                + (np.sum([np.sqrt(randomFieldV.eigenvalues[m]) * 3 * np.max([np.abs(randomFieldV.eigenvectors[randomFieldV.N + j, m]) for j in range(randomFieldV.N)]) for m in range(len_xi)]))**2)
    print(f"R_nv (maximal perturbation distance): {R_nv}") # maximal perturbation distance

    # Non-Varying Area
    if R_nv >= 1:
            raise ValueError(f"for len(xi): {len_xi} the non-varying area is the empty set")
    return mshr.Circle(CENTER, 1 - R_nv)
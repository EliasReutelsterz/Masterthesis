import matplotlib.pyplot as plt
import fenics as fe
import mshr
import numpy as np
from scipy.linalg import eig
import sympy as sp
from scipy.interpolate import griddata
import plotly.graph_objects as go



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

class RandomFieldV():
    def __init__(self, eigenvalues, eigenvectors, basis_functions, N, J):
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

    #! triangle areas not needed it cancels out, weight just times 2
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
    def __init__(self, eigenvalues, eigenvectors, basis_functions: list[ConstBasisFunction], N: int, J: int):
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
    def __init__(self, randomfieldV, xi, **kwargs):
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
    


def solve_poisson_for_given_sample(mesh_resolution, jacobianV, xi, f):
    jacobianV_fixed_xi = JacobianVFixedXi(jacobianV, xi)
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 3)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    A_expr = AExpression(jacobianV_fixed_xi, degree=2)
    a = fe.inner(fe.dot(A_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV_fixed_xi, degree=2)
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
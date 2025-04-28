import fenics as fe
import numpy as np
from scipy.linalg import eig
import mshr
import os
import time
import sympy as sp
import matplotlib.pyplot as plt


# factors involved in the covariance
alpha = 1E3
beta = 10


# General helpers

x1, x2, y1, y2 = sp.symbols('x1 x2 y1 y2')

v_cov1_1_sym = alpha * 5.0/100.0 * sp.exp(beta * -4.0 * ((x1 - y1)**2 + (x2 - y2)**2))
v_cov1_2_sym = alpha * 1.0/100.0 * sp.exp(-0.1 * ((2*x1 - y1)**2 + (2*x2 - y2)**2))
v_cov2_1_sym = alpha * 1.0/100.0 * sp.exp(-0.1 * ((x1 - 2*y1)**2 + (x2 - 2*y2)**2))
v_cov2_2_sym = alpha * 5.0/100.0 * sp.exp(beta * -1.0 * ((x1 - y1)**2 + (x2 - y2)**2))

v_cov1_1 = sp.lambdify((x1, x2, y1, y2), v_cov1_1_sym, 'numpy')
v_cov1_2 = sp.lambdify((x1, x2, y1, y2), v_cov1_2_sym, 'numpy')
v_cov2_1 = sp.lambdify((x1, x2, y1, y2), v_cov2_1_sym, 'numpy')
v_cov2_2 = sp.lambdify((x1, x2, y1, y2), v_cov2_2_sym, 'numpy')

class Quad_point():
    def __init__(self, point: list, weight: float):
        self.point = point
        self.weight = weight

QUAD_POINTS_2DD_6 = [Quad_point([(6 - np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(9 + 2 * np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(6 - np.sqrt(15)) / 21, (9 + 2 * np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(6 + np.sqrt(15)) / 21, (9 - 2 * np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([(6 + np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([(9 - 2 * np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([1 / 3, 1 / 3], 9/80)]

def find_affine_transformation(triangle):
    transformation_matrix = np.array([[triangle[1, 0] - triangle[0, 0], triangle[2, 0] - triangle[0, 0]],
                                    [triangle[1, 1] - triangle[0, 1], triangle[2, 1] - triangle[0, 1]]])
    transformation_vector = np.array([triangle[0, 0], triangle[0, 1]])
    return transformation_matrix, transformation_vector

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


# Random field VBar
# Uses centered circle with radius 0.02 as domain

class RandomFieldVBar():
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
    
def get_C_entry_randomFieldVBar(f, basis_function_i: ConstBasisFunction, basis_function_j: ConstBasisFunction):
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

def calculate_randomFieldVBar(mesh_resolution):
    domain = mshr.Circle(fe.Point(0.0, 0.0), 0.02) 
    mesh = mshr.generate_mesh(domain, mesh_resolution)
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
                C[i, j] = C[j, i] = get_C_entry_randomFieldVBar(v_cov1_1, basis_function_i, basis_function_j)
                C[i, N + j] = C[j, N + i] = get_C_entry_randomFieldVBar(v_cov1_2, basis_function_i, basis_function_j)
                C[N + i, j] = C[N + j, i] = get_C_entry_randomFieldVBar(v_cov2_1, basis_function_i, basis_function_j)
                C[N + i, N + j] = C[N + j, N + i] = get_C_entry_randomFieldVBar(v_cov2_2, basis_function_i, basis_function_j)

    M = np.zeros((2 * N, 2 * N))
    for i, basis_function_i in enumerate(basis_functions):
        integrand = basis_function_i.function * basis_function_i.function * fe.dx
        M[i, i] = M[N + i, N + i] = fe.assemble(integrand)

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
    return RandomFieldVBar(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J)
    

# Random field V

def perturbation_function(x: np.array, omega2: np.array, r: float, randomFieldVBar: RandomFieldVBar) -> np.array:
    x = x - np.array([0.16, 0.16])
    c = np.sqrt(x[0]**2 + x[1]**2)
    x_circ_proj = r/c * x

    theta = np.arctan2(x[1], x[0]) # order has to be y, x as tan(y/x)=theta

    if -np.pi/4 <= theta <= np.pi/4:
        x_bound_proj = np.array([0.16, 0.16*np.tan(theta)])
    elif np.pi/4 <= theta <= 3*np.pi/4:
        x_bound_proj = np.array([0.16 / np.tan(theta), 0.16])
    elif theta <= -3*np.pi/4 or theta >= 3*np.pi/4:
        x_bound_proj = np.array([-0.16, -0.16*np.tan(theta)])
    else:
        x_bound_proj = np.array([-0.16 / np.tan(theta), -0.16])

    h_max = np.sqrt((x_bound_proj[0] - x_circ_proj[0])**2 + (x_bound_proj[1] - x_circ_proj[1])**2)
    h = np.sqrt((x[0] - x_bound_proj[0])**2 + (x[1] - x_bound_proj[1])**2)

    bound_perturb = x_bound_proj

    circ_perturb = np.array(randomFieldVBar(x_circ_proj, omega2))

    x_pert = h / h_max * circ_perturb + (1 - h / h_max) * bound_perturb

    return np.array([0.16, 0.16]) + x_pert

def perturb_mesh(mesh: fe.Mesh, omega2: np.array, r: float, randomFieldVBar: RandomFieldVBar) -> fe.Mesh:
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega2, r, randomFieldVBar)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh

class RandomFieldVExpression(fe.UserExpression):
    def __init__(self, r: float, omega2: np.array, domain: fe.Mesh, randomFieldVBar: RandomFieldVBar, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.omega2 = omega2
        self.domain = domain
        self.randomFieldVBar = randomFieldVBar

    def eval(self, values, x):
        perturbed_point = perturbation_function(x=x, omega2=self.omega2, r=self.r, randomFieldVBar=self.randomFieldVBar)
        values[0] = perturbed_point[0]
        values[1] = perturbed_point[1]

    def value_shape(self):
        return (2,)
    
class J_minus_TExpression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        J_inv_T = np.linalg.inv(J).T
        values[0] = J_inv_T[0, 0]
        values[1] = J_inv_T[0, 1]
        values[2] = J_inv_T[1, 0]
        values[3] = J_inv_T[1, 1]

    def value_shape(self):
        return (2, 2)
    
class J_helper1Expression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = J[1, 1]
        values[1] = - J[1, 0]

    def value_shape(self):
        return (2, )

class J_helper2Expression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = - J[0, 1]
        values[1] = J[1, 1]

    def value_shape(self):
        return (2, )
    
class J_helper3Expression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = np.sqrt(J[0, 0]**2 + J[0, 1]**2)

    def value_shape(self):
        return ()

class J_determinantExpression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J_det = np.linalg.det(self.jacobianProj(x).reshape((2, 2)))
        if J_det == 0:
            print("Determinant 0.")
            values[0] = 1
        else:
            values[0] = J_det

    def value_shape(self):
        return ()
    
class J_inv_determinantExpression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J_det = np.linalg.det(self.jacobianProj(x).reshape((2, 2)))
        if J_det == 0:
            print("Determinant 0.")
            values[0] = 1
        else:
            values[0] = 1 / J_det

    def value_shape(self):
        return ()
    

# Random field E

class RandomFieldE():
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array, basis_functions: list[ConstBasisFunction], N: int, J: int, mu: float):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J
        self.mu = mu

    def __call__(self, x, xi):
        index_supported_basis_function = self.find_supported_basis_function(x)
        return np.exp(self.mu + sum([np.sqrt(self.eigenvalues[m]) * self.eigenvectors[index_supported_basis_function, m] for m in range(len(xi))]))
    
    def find_supported_basis_function(self, x):
        for i, basis_function in enumerate(self.basis_functions):
            if basis_function.function(x) == 1:
                return i
        raise ValueError("No supported basis function found for x")

class RandomFieldEExpression(fe.UserExpression):
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, omega2: np.array, r: float, randomFieldVBar: RandomFieldVBar, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi
        self.omega2 = omega2
        self.r = r
        self.randomFieldVBar = randomFieldVBar

    def eval(self, values, x):
        x_pert = perturbation_function(x=x, omega2=self.omega2, r=self.r, randomFieldVBar=self.randomFieldVBar)
        values[0] = self.randomFieldE(x_pert, self.xi)
    
    def value_shape(self):
        return ()

class RandomFieldEHatExpression(fe.UserExpression):
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, omega2: np.array, r: float, randomFieldVBar: RandomFieldVBar, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi
        self.omega2 = omega2
        self.r = r
        self.randomFieldVBar = randomFieldVBar

    def eval(self, values, x):
        x_pert = perturbation_function(x=x, omega2=self.omega2, r=self.r, randomFieldVBar=self.randomFieldVBar)
        values[0] = self.randomFieldE(x_pert, self.xi)
    
    def value_shape(self):
        return ()

def matern_covariance_function(sigma: float, x: np.array, y: np.array) -> float:
    # nu = 1/2
    l = 0.02
    h = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    return sigma**2 * np.exp(-h/l)

def get_C_entry_randomFieldE(sigma: float, basis_function_i: ConstBasisFunction, basis_function_j: ConstBasisFunction) -> float:
    transformation_matrix_x, transformation_vector_x = find_affine_transformation(basis_function_i.vertex_coords)
    transformation_matrix_y, transformation_vector_y = find_affine_transformation(basis_function_j.vertex_coords)

    active_quad_points = QUAD_POINTS_2DD_6

    quad_points_x = [Quad_point(np.dot(transformation_matrix_x, quad_point_x.point) + transformation_vector_x, quad_point_x.weight * 2 * basis_function_i.triangle_area) for quad_point_x in active_quad_points]
    quad_points_y = [Quad_point(np.dot(transformation_matrix_y, quad_point_y.point) + transformation_vector_y, quad_point_y.weight * 2 * basis_function_j.triangle_area) for quad_point_y in active_quad_points]
    integral = 0
     
    for quad_point_x in quad_points_x:
        for quad_point_y in quad_points_y:
            integral += matern_covariance_function(sigma, quad_point_x.point, quad_point_y.point) * quad_point_x.weight * quad_point_y.weight
    return integral

def calculate_randomFieldE(mesh_resolution: int) -> RandomFieldE:
    
    # Parameters
    mu = 26.011
    sigma = 0.149

    # Mesh
    domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) # used a_plate_length hardcoded
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = 0.02 # used r hardcoded
    domain = domain - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    
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

    C = np.zeros((N, N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if j <= i:
                # Here we use that each block is symmetric because of the symmetry of the covariance functions
                C[i, j] = C[j, i] = get_C_entry_randomFieldE(sigma, basis_function_i, basis_function_j)

    M = np.zeros((N, N))
    for i, basis_function_i in enumerate(basis_functions):
        integrand = basis_function_i.function * basis_function_i.function * fe.dx
        M[i, i] = fe.assemble(integrand)

    J = N
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

    return RandomFieldE(eigenvalues=sorted_eigenvalues,
                        eigenvectors=sorted_eigenvectors,
                        basis_functions=basis_functions,
                        N=N,
                        J=J,
                        mu=mu)
 

# Model

def sample_omega1(randomFieldE: RandomFieldE) -> np.array:
    return np.random.normal(0, 1, size=randomFieldE.J)

def sample_omega2(randomFieldVBar: RandomFieldVBar) -> np.array:
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=randomFieldVBar.J)

def sample_q() -> float:
    mu_q = 60*1E6
    sigma_q = 12*1E6
    return np.random.normal(mu_q, sigma_q)

def euclidean_distance(x, y):
        return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def create_mesh(mesh_resolution: int) -> fe.Mesh:
    domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) # used a_plate_length hardcoded
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = 0.02 # used r hardcoded
    domain = domain - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    return mesh

def solve_model(mesh_resolution: int, omega1: np.array, omega2: np.array, q: float, randomFieldE: RandomFieldE = None, randomFieldVBar: RandomFieldVBar = None) -> fe.Function:

    # Model parameters
    a_plate_length = 0.32
    r = 0.02
    nu = 0.29
    rho = 7850
    grav = 9.80665
    b = fe.Constant((0, -rho * grav))

    # Mesh
    mesh = create_mesh(mesh_resolution=mesh_resolution)
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = r
    first_cell = fe.Cell(mesh, 0)
    vertex_length = first_cell.h()

    # mark all vertices
    boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)

    # mark inner circle boundary
    inner_circle_boundary_points = []
    class InnerCircleBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            dist_to_center = euclidean_distance(x, circ_center)
            if on_boundary and dist_to_center < circ_radius + vertex_length/2:
                inner_circle_boundary_points.append(np.array([x[0], x[1]]))
                return True
            else:
                return False
    inner_circle_boundary = InnerCircleBoundary()
    inner_circle_boundary.mark(boundary_markers, 1)

    # mark left boundary
    left_boundary_points = []
    class LeftBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary and fe.near(x[0], 0):
                left_boundary_points.append(np.array([x[0], x[1]]))
                return True
            else:
                return False
    left_boundary = LeftBoundary()
    left_boundary.mark(boundary_markers, 2)

    def left_boundary_function(x, on_boundary):
        return on_boundary and fe.near(x[0], 0)

    # mark right boundary
    right_boundary_points = []
    class RightBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary and fe.near(x[0], 0.32):
                right_boundary_points.append(np.array([x[0], x[1]]))
                return True
            else:
                return False
    right_boundary = RightBoundary()
    right_boundary.mark(boundary_markers, 3)

    # Extract the boundary points
    inner_circle_boundary_points = np.array(inner_circle_boundary_points)
    left_boundary_points = np.array(left_boundary_points)
    right_boundary_points = np.array(right_boundary_points)
    
    # Random field V
    if not randomFieldVBar:
        randomFieldVBar = calculate_randomFieldVBar(mesh_resolution=mesh_resolution)
    randomFieldVExpression = RandomFieldVExpression(omega2=omega2, r=r, domain=mesh, randomFieldVBar=randomFieldVBar)

    # Random field Ê and E
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEHatExpression = RandomFieldEHatExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2, r=r, randomFieldVBar=randomFieldVBar)


    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    u_hat = fe.TrialFunction(V)
    v_hat = fe.TestFunction(V)
    u_hat_sol = fe.Function(V)

    # Boundary conditions
    ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    bc_left = fe.DirichletBC(V, fe.Constant((0,0)), left_boundary_function)
    g = fe.Constant((q,0))


    left_prefactor = fe.Expression('E / (2 * (1 + nu))', degree=1, E=randomFieldEHatExpression, nu=nu, domain=mesh)
    right_prefactor = fe.Expression('E / (2 * (1 - nu))', degree=1, E=randomFieldEHatExpression, nu=nu, domain=mesh)


    randomFieldEProj = fe.project(randomFieldVExpression, V)
    W = fe.TensorFunctionSpace(mesh, 'P', 1)
    jacobianProj = fe.project(fe.grad(randomFieldEProj), W)

    J_inv_T = J_minus_TExpression(jacobianProj, domain=mesh)
    J_helper1 = J_helper1Expression(jacobianProj, domain=mesh)
    J_helper2 = J_helper2Expression(jacobianProj, domain=mesh)
    det_J = J_determinantExpression(jacobianProj, domain=mesh)
    inv_det_J = J_inv_determinantExpression(jacobianProj, domain=mesh)
    J_helper3 = J_helper3Expression(jacobianProj, domain=mesh)


    left_integrand = det_J * fe.inner(fe.dot(J_inv_T, fe.grad(u_hat)), fe.dot(J_inv_T, fe.grad(left_prefactor * v_hat)))

    G_hat = fe.as_matrix([[fe.dot(J_helper1, fe.grad(u_hat)[:, 0]) + (2 * nu)/(1 + nu) * fe.dot(J_helper2, fe.grad(u_hat)[:, 1]),
                        (1 - nu)/(1 + nu) * fe.dot(J_helper2, fe.grad(u_hat)[:, 0])],
                        [(1 - nu)/(1 + nu) * fe.dot(J_helper1, fe.grad(u_hat)[:, 1]),
                            (2 * nu)/(1 + nu) * fe.dot(J_helper1, fe.grad(u_hat)[:, 0]) + fe.dot(J_helper2, fe.grad(u_hat)[:, 1])]])

    right_integrand = fe.inner(G_hat, fe.dot(J_inv_T, fe.grad(right_prefactor * v_hat)))


    a = (left_integrand + right_integrand) * fe.dx
    # Right-hand side of weak form
    L = det_J * fe.dot(b,v_hat) * fe.dx + J_helper3 * fe.dot(g,v_hat) * ds(3)

    # Solve Galerkin system
    fe.solve(a==L, u_hat_sol, bc_left)
    
    def sigma_hat(u_hat):
        E_hat = randomFieldEHatExpression
        u_hat_grad = fe.grad(u_hat)
        entry_11 = 1/(1 - nu**2) * (fe.dot(J_helper1, u_hat_grad[:, 0]) + nu * fe.dot(J_helper2, u_hat_grad[:, 1]))
        entry_22 = 1/(1 - nu**2) * (fe.dot(J_helper2, u_hat_grad[:, 1]) + nu * fe.dot(J_helper1, u_hat_grad[:, 0]))
        entry_12 = 1/(2*(1 + nu)) * (fe.dot(J_helper1, u_hat_grad[:, 1]) + fe.dot(J_helper2, u_hat_grad[:, 0]))
        return E_hat * inv_det_J * fe.as_matrix([[entry_11, entry_12], [entry_12, entry_22]])


    sigma_hat_proj = fe.project(sigma_hat(u_hat_sol), fe.TensorFunctionSpace(mesh, 'P', 1))


    u_hat_sol_data = u_hat_sol.vector()[:]
    sigma_hat_proj_data = sigma_hat_proj.vector()[:]

    return u_hat_sol_data, sigma_hat_proj_data

def solve_model_with_plots(mesh_resolution: int, omega1: np.array, omega2: np.array, q: float, randomFieldE: RandomFieldE = None, randomFieldVBar: RandomFieldVBar = None) -> None:

    # Model parameters
    a_plate_length = 0.32
    r = 0.02
    nu = 0.29
    rho = 7850
    grav = 9.80665
    b = fe.Constant((0, -rho * grav))

    # Mesh
    bottom_left_corner = fe.Point(0, 0)
    top_right_corner = fe.Point(a_plate_length, a_plate_length)
    domain = mshr.Rectangle(bottom_left_corner, top_right_corner)
    circ_center = fe.Point((top_right_corner[0] + bottom_left_corner[0])/2, (top_right_corner[1] + bottom_left_corner[1])/2)
    circ_radius = r
    domain = domain - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    first_cell = fe.Cell(mesh, 0)
    vertex_length = first_cell.h()

    # mark all vertices
    boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)

    # mark inner circle boundary
    inner_circle_boundary_points = []
    class InnerCircleBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            dist_to_center = euclidean_distance(x, circ_center)
            if on_boundary and dist_to_center < circ_radius + vertex_length/3:
                inner_circle_boundary_points.append(np.array([x[0], x[1]]))
                return True
            else:
                return False
    inner_circle_boundary = InnerCircleBoundary()
    inner_circle_boundary.mark(boundary_markers, 1)

    # mark left boundary
    left_boundary_points = []
    class LeftBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary and fe.near(x[0], bottom_left_corner[0]):
                left_boundary_points.append(np.array([x[0], x[1]]))
                return True
            else:
                return False
    left_boundary = LeftBoundary()
    left_boundary.mark(boundary_markers, 2)

    def left_boundary_function(x, on_boundary):
        return on_boundary and fe.near(x[0], bottom_left_corner[0])

    # mark right boundary
    right_boundary_points = []
    class RightBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            if on_boundary and fe.near(x[0], top_right_corner[0]):
                right_boundary_points.append(np.array([x[0], x[1]]))
                return True
            else:
                return False
    right_boundary = RightBoundary()
    right_boundary.mark(boundary_markers, 3)

    # Extract the boundary points
    inner_circle_boundary_points = np.array(inner_circle_boundary_points)
    left_boundary_points = np.array(left_boundary_points)
    right_boundary_points = np.array(right_boundary_points)


    # Random field V
    if not randomFieldVBar:
        randomFieldVBar = calculate_randomFieldVBar(mesh_resolution=mesh_resolution)
    perturbed_mesh = perturb_mesh(mesh=mesh, omega=omega2, r=r, randomFieldVBar=randomFieldVBar)
    randomFieldVExpression = RandomFieldVExpression(r=r, omega2=omega2, domain=mesh, randomFieldVBar=randomFieldVBar)

    # Random field Ê and E
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEHatExpression = RandomFieldEHatExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2, r=r, randomFieldVBar=randomFieldVBar)
    randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2, r=r, randomFieldVBar=randomFieldVBar)


    # deg = 1
    # V = fe.VectorFunctionSpace(mesh, 'P', deg)

    deg_test = 1
    V_test = fe.VectorFunctionSpace(mesh, 'P', deg_test)
    randomFieldVProj_test = fe.project(randomFieldVExpression, V_test)

    W_test = fe.TensorFunctionSpace(mesh, 'P', deg_test)
    jacobianProj = fe.project(fe.grad(randomFieldVProj_test), W_test)

    J_helper1 = J_helper1Expression(jacobianProj, domain=mesh)
    J_helper2 = J_helper2Expression(jacobianProj, domain=mesh)
    inv_det_J = J_inv_determinantExpression(jacobianProj, domain=mesh)
    
    u_hat_sol_data, _ = solve_model(mesh_resolution, omega1, omega2, q, randomFieldE)

    u_hat_sol = fe.Function(V_test)
    u_hat_sol.vector()[:] = u_hat_sol_data

    # Plots
    plt.figure(figsize=(16, 8))

    # Plot the mesh and boundary points
    plt.subplot(2, 4, 1)
    fe.plot(mesh, title=r'Reference Mesh for $\Omega_{ref}$')
    plt.scatter(inner_circle_boundary_points[:, 0], inner_circle_boundary_points[:, 1], color='cyan', s=10, label='Circle Boundary Points')
    plt.scatter(left_boundary_points[:, 0], left_boundary_points[:, 1], color='blue', s=10, label='Left Boundary Points')
    plt.scatter(right_boundary_points[:, 0], right_boundary_points[:, 1], color='green', s=10, label='Right Boundary Points')
    plt.xlabel(r'$\hat{x}_1$')
    plt.ylabel(r'$\hat{x}_2$')
    plt.legend(loc='upper right')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    # Mark inner circle boundary points for perturbed mesh
    perturbed_inner_circle_boundary_points = []
    for point in inner_circle_boundary_points:
        perturbed_inner_circle_boundary_points.append(perturbation_function(x=point, omega2=omega2, r=r, randomFieldVBar=randomFieldVBar))
    perturbed_inner_circle_boundary_points = np.array(perturbed_inner_circle_boundary_points)

    # Plot perturbed mesh
    plt.subplot(2, 4, 5)
    fe.plot(perturbed_mesh, title=r'Perturbed Mesh for $\Omega(\omega2)$')
    plt.scatter(left_boundary_points[:, 0], left_boundary_points[:, 1], color='blue', s=10, label='Left Boundary Points')
    plt.scatter(right_boundary_points[:, 0], right_boundary_points[:, 1], color='green', s=10, label='Right Boundary Points')
    plt.scatter(perturbed_inner_circle_boundary_points[:, 0], perturbed_inner_circle_boundary_points[:, 1], color='cyan', s=10, label='Circle Boundary Points')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper right')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    # Plot the solution û
    plt.subplot(2, 4, 2)
    c = fe.plot(u_hat_sol, title=r'Displacement $\hat{u}(\hat{x}, \omega)$')
    plt.colorbar(c)
    plt.title(r'Displacement $\hat{u}(\hat{x}, \omega)$')
    plt.xlabel(r'$\hat{x}_1$')
    plt.ylabel(r'$\hat{x}_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    # Plot the solution u
    V_perturbed = fe.VectorFunctionSpace(perturbed_mesh, "P", 1)
    u_sol = fe.Function(V_perturbed)
    u_sol.vector()[:] = u_hat_sol.vector()[:] 
    plt.subplot(2, 4, 6)
    c = fe.plot(u_sol, title=r'Displacement $u(x, \omega)$')
    plt.colorbar(c)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    print(f"q: {q}")

    # Plot Sigma(u)
    def sigma(u): # Stress tensor
        E = randomFieldEExpression
        u_grad = fe.grad(u)
        matrix_expr = fe.as_matrix([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        epsilon = fe.as_vector([u_grad[0, 0], u_grad[1, 1], u_grad[0, 1] + u_grad[1, 0]])
        sigma_vectorized = E / (1 - nu**2) * fe.dot(matrix_expr, epsilon)
        return fe.as_matrix([[sigma_vectorized[0], sigma_vectorized[2]], [sigma_vectorized[2], sigma_vectorized[1]]])

    def sigma_hat(u_hat):
        E_hat = randomFieldEHatExpression
        u_hat_grad = fe.grad(u_hat)
        entry_11 = 1/(1 - nu**2) * (fe.dot(J_helper1, u_hat_grad[:, 0]) + nu * fe.dot(J_helper2, u_hat_grad[:, 1]))
        entry_22 = 1/(1 - nu**2) * (fe.dot(J_helper2, u_hat_grad[:, 1]) + nu * fe.dot(J_helper1, u_hat_grad[:, 0]))
        entry_12 = 1/(2*(1 + nu)) * (fe.dot(J_helper1, u_hat_grad[:, 1]) + fe.dot(J_helper2, u_hat_grad[:, 0]))
        return  E_hat * inv_det_J * fe.as_matrix([[entry_11, entry_12], [entry_12, entry_22]])

    sigma_hat_proj = fe.project(sigma_hat(u_hat_sol)[:, 0], V_test)
    # for right_boundary_point in right_boundary_points:
    #     print(f"sigma_hat_proj({right_boundary_point}): {sigma_hat_proj(right_boundary_point)}")

    plt.subplot(2, 4, 3)
    c = fe.plot(sigma_hat_proj, title=r'$\hat{\sigma}(\hat{u}) \cdot e_1$')
    plt.colorbar(c)
    plt.xlabel(r'$\hat{x}_1$')
    plt.ylabel(r'$\hat{x}_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)


    # Plot Sigma(u)
    sigma_proj = fe.project(sigma(u_sol)[:, 0], V_perturbed)
    for right_bounday_point in right_boundary_points:
        print(f"sigma_proj({right_bounday_point}): {sigma_proj(right_bounday_point)}")

    plt.subplot(2, 4, 7)
    c = fe.plot(sigma_proj, title=r'$\sigma(u) \cdot e_1$')
    plt.colorbar(c)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    # Plot random field E(V(^x))
    W = fe.FunctionSpace(mesh, 'P', 1)
    E_proj = fe.project(randomFieldEHatExpression, W)
    plt.subplot(2, 4, 4)
    c = fe.plot(E_proj, title=r"Random Field $E(V(\hat{x}, \omega2), \omega1)$")
    plt.colorbar(c)
    plt.xlabel(r'$\hat{x}_1$')
    plt.ylabel(r'$\hat{x}_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    # Plot random field E(x)
    W_perturbed = fe.FunctionSpace(perturbed_mesh, "P", 1)
    E_perturbed_proj = fe.Function(W_perturbed)
    E_perturbed_proj.assign(fe.project(randomFieldEExpression, W_perturbed))
    plt.subplot(2, 4, 8)
    c = fe.plot(E_perturbed_proj, title=r"Random Field $E(x, \omega1)$")
    plt.colorbar(c)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    plt.tight_layout()
    plt.show()




    # Plot sigma(u) and sigma_hat(u_hat) along right boundary
    right_boundary_points = np.linspace(start=0, stop=0.32, num=100)
    sigma_u_right_boundary = np.array([sigma_proj(np.array([0.32, point])) for point in right_boundary_points])
    

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(right_boundary_points, sigma_u_right_boundary[:, 0], label=r'First component $\sigma(u) \cdot e_1$')
    plt.axhline(q, label='Reference for $g[0]=q$', color="red")
    plt.legend()
    plt.xlabel(r'$x_2$')
    
    plt.subplot(1, 2, 2)
    plt.plot(right_boundary_points, sigma_u_right_boundary[:, 1], label=r'Second component $\sigma(u) \cdot e_1$')
    plt.axhline(0, label='Reference for $g[1]=0$', color="red")
    plt.xlabel(r'$x_2$')
    plt.legend()
    plt.tight_layout()
    plt.show()


    sigma_hat_u_hat_right_boundary = np.array([sigma_hat_proj(np.array([0.32, point])) for point in right_boundary_points])
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(right_boundary_points, sigma_hat_u_hat_right_boundary[:, 0], label=r'First component $\hat{\sigma}(\hat{u}) \cdot e_1$')
    plt.axhline(q, label='Reference for $g[0]=q$', color="red")
    plt.legend()
    plt.xlabel(r'$\hat{x}_2$')
    
    plt.subplot(1, 2, 2)
    plt.plot(right_boundary_points, sigma_hat_u_hat_right_boundary[:, 1], label=r'Second component $\hat{\sigma}(\hat{u}) \cdot e_1$')
    plt.axhline(0, label='Reference for $g[1]=0$', color="red")
    plt.legend()
    plt.xlabel(r'$\hat{x}_2$')
    plt.tight_layout()
    plt.show()

    return None


# Monte Carlo

#! Still to be tested!

def save_mc_samples(u_hat_sols: np.array, mesh_resolution_kl_v: int, mesh_resolution_kl_e: int, mesh_resolution: int) -> None:
    base_path = f'mc_data_storage/klresv_{mesh_resolution_kl_v}_klrese_{mesh_resolution_kl_e}_femres_{mesh_resolution}'
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, 'u_hat_sols.npy')
    if os.path.exists(file_path):
        u_hat_sols_existing = np.load(file_path)
        u_hat_sols = np.concatenate((u_hat_sols_existing, u_hat_sols), axis=0)
    np.save(file_path, u_hat_sols)

def run_and_save_mc(mesh_resolution_kl_v: int, mesh_resolution_kl_e: int, mesh_resolution: int, sample_size: int, randomFieldVBar: RandomFieldVBar = None, randomFieldE: RandomFieldE = None) -> None:
    if randomFieldVBar is None:
        randomFieldVBar = calculate_randomFieldVBar(mesh_resolution=mesh_resolution_kl_v)
    if randomFieldE is None:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution_kl_e)
    mesh = create_mesh(mesh_resolution=mesh_resolution)
    u_hat_sols = np.zeros((sample_size, mesh.coordinates().shape[0], 2))
    for i in range(sample_size):
        omega1 = sample_omega1(randomFieldE)
        omega2 = sample_omega2(randomFieldVBar)
        q = sample_q()
        u_hat_sols[i] = solve_model(mesh_resolution=mesh_resolution, omega1=omega1, omega2=omega2, q=q, randomFieldE=randomFieldE, randomFieldVBar=randomFieldVBar).vector().get_local().reshape((-1, 2))
    save_mc_samples(u_hat_sols, mesh_resolution_kl_v, mesh_resolution_kl_e, mesh_resolution)

def mc_analysis(sparse_mesh_resolution_kl_v: int, sparse_mesh_resolution_kl_e: int,
                fine_mesh_resolution_kl_v: int, fine_mesh_resolution_kl_e: int,
                mesh_resolution: int, P_hat: fe.Point) -> None:
    
    sparse_u_hat_sols = np.load(f'mc_data_storage/klresv_{sparse_mesh_resolution_kl_v}_klrese_{sparse_mesh_resolution_kl_e}_femres_{mesh_resolution}/u_hat_sols.npy')
    fine_u_hat_sols = np.load(f'mc_data_storage/klresv_{fine_mesh_resolution_kl_v}_klrese_{fine_mesh_resolution_kl_e}_femres_{mesh_resolution}/u_hat_sols.npy')

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [4]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= sparse_u_hat_sols.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------

    # Mean and variance of fine solutions
    V = fe.VectorFunctionSpace(create_mesh(mesh_resolution), 'P', 1)
    fine_mean_u_hat_sols = fe.Function(V)
    fine_var_u_hat_sols = fe.Function(V)

    fine_mean_u_hat_sols_array = np.mean(fine_u_hat_sols, axis=0).reshape(-1)
    fine_var_u_hat_sols_array = np.var(fine_u_hat_sols, axis=0).reshape(-1)

    fine_mean_u_hat_sols.vector()[:] = fine_mean_u_hat_sols_array
    fine_var_u_hat_sols.vector()[:] = fine_var_u_hat_sols_array
    
    c = fe.plot(fine_mean_u_hat_sols, title='Mean of fine u_hat_sols')
    plt.colorbar(c)
    plt.show()

    c = fe.plot(fine_var_u_hat_sols[0], title='Variance of first component of fine u_hat_sols')
    plt.colorbar(c)
    plt.show()

    # Sparse solutions analysis

    sparse_u_hat_sols_functions = []
    sparse_u_hat_sols_P_hat_means = []
    L2_errors = []
    H1_errors = []
    for mc_sample_index, mc_sample_size in enumerate(mc_sample_sizes):
        sparse_mean_sol = fe.Function(V)
        sparse_mean_sol.set_allow_extrapolation(True)

        # Calculate mean solution
        if mc_sample_index == 0:
            data_sparse_sample = sparse_u_hat_sols[:mc_sample_size]
        else:
            data_sparse_sample = sparse_u_hat_sols[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]
        sparse_mean_sol.vector()[:] = np.mean(data_sparse_sample, axis=0).reshape(-1)
        sparse_u_hat_sols_functions.append(sparse_mean_sol)
        sparse_u_hat_sols_P_hat_means.append(sparse_mean_sol(P_hat))
        
        # Calculate L2 and H1 errors
        L2_errors.append(fe.errornorm(fine_mean_u_hat_sols, sparse_mean_sol, 'L2'))
        H1_errors.append(fe.errornorm(fine_mean_u_hat_sols, sparse_mean_sol, 'H1'))

    # Plot means in P_hat
    mean_errors_P_hat = []
    for i in range(len(mc_sample_sizes)):
        mean_errors_P_hat.append(euclidean_distance(sparse_u_hat_sols_P_hat_means[i], fine_mean_u_hat_sols(P_hat)))
    fig = plt.figure(figsize=(10, 4))
    plt.plot(mc_sample_sizes, mean_errors_P_hat, 'bo', marker='x', linestyle='None')
    plt.xscale('log')
    plt.xlabel('MC Samples')
    plt.ylabel('Error')
    plt.legend(loc='upper left')
    plt.title(fr"Euclidean error to mean reference solution in $\hat{{P}} = ({P_hat[0]}, {P_hat[1]})$")
    plt.grid(True)
    plt.show()

    # Plots L2 and H1 errors
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


# Sobol Indices

def get_quadrature_basis_functions(mesh: fe.Mesh):
    V = fe.FunctionSpace(mesh, 'DG', 0)
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
    return basis_functions

def sp_6_sobol_run_samples_and_save(mc_sample_size: int, fem_res: int, kl_res_e: int, kl_res_v: int, size_of_xi_e: int, size_of_xi_v: int, randomFieldE: RandomFieldE = None, randomFieldVBar: RandomFieldVBar = None) -> None:

    if randomFieldE is None:
        randomFieldE = calculate_randomFieldE(kl_res_e)

    if randomFieldVBar is None:
        randomFieldVBar = calculate_randomFieldVBar(mesh_resolution=kl_res_v)
    
    mesh = create_reference_mesh(fem_res)
    quadrature_basis_functions = get_quadrature_basis_functions(mesh)
    N = len(quadrature_basis_functions)

    total_num_analyzed_rvs = size_of_xi_e + size_of_xi_v + 1
    total_num_rvs = randomFieldE.J + randomFieldVBar.J + 1

    A = np.zeros((mc_sample_size, total_num_rvs))
    B = np.zeros((mc_sample_size, total_num_rvs))

    A[:, :randomFieldE.J] = np.array([sample_omega1(randomFieldE) for _ in range(mc_sample_size)])
    B[:, :randomFieldE.J] = np.array([sample_omega1(randomFieldE) for _ in range(mc_sample_size)])
    A[:, randomFieldE.J:randomFieldE.J + randomFieldVBar.J] = np.array([sample_omega2(randomFieldVBar) for _ in range(mc_sample_size)])
    B[:, randomFieldE.J:randomFieldE.J + randomFieldVBar.J] = np.array([sample_omega2(randomFieldVBar) for _ in range(mc_sample_size)])
    A[:, randomFieldE.J + randomFieldVBar.J] = np.array([sample_q() for _ in range(mc_sample_size)])
    B[:, randomFieldE.J + randomFieldVBar.J] = np.array([sample_q() for _ in range(mc_sample_size)])

    mesh = create_reference_mesh(fem_res)
    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    W = fe.TensorFunctionSpace(mesh, 'P', 1)
    tensor_function = fe.Function(W)
    M = tensor_function.vector().size()

    sigma_hat_f_A = np.zeros((mc_sample_size, M))
    u_hat_f_A = np.zeros((mc_sample_size, N, 2))
    sigma_hat_f_B = np.zeros((mc_sample_size, M))
    u_hat_f_B = np.zeros((mc_sample_size, N, 2))

    for m in range(mc_sample_size):
        print(f"First loop Iteration {m+1}/{mc_sample_size}")
        u_sol_A_data, sigma_hat_A_data = solve_model(fem_res, A[m, :randomFieldE.J], A[m, randomFieldE.J:randomFieldE.J + randomFieldVBar.J], A[m, randomFieldE.J + randomFieldVBar.J], randomFieldE, randomFieldVBar)
        u_sol_A = fe.Function(V)
        u_sol_A.vector()[:] = u_sol_A_data
        sigma_hat_f_A[m] = sigma_hat_A_data
        u_sol_B_data, sigma_hat_B_data = solve_model(fem_res, B[m, :randomFieldE.J], B[m, randomFieldE.J:randomFieldE.J + randomFieldVBar.J], B[m, randomFieldE.J + randomFieldVBar.J], randomFieldE, randomFieldVBar)
        u_sol_B = fe.Function(V)
        u_sol_B.vector()[:] = u_sol_B_data
        sigma_hat_f_B[m] = sigma_hat_B_data

        for basis_function_index, basis_function in enumerate(quadrature_basis_functions):
            transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
            active_quad_points = QUAD_POINTS_2DD_6
            quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
            int_A = 0
            int_B = 0
            for quad_point in quad_points:
                int_A += u_sol_A(quad_point.point) * quad_point.weight
                int_B += u_sol_B(quad_point.point) * quad_point.weight
            u_hat_f_A[m, basis_function_index, :] = int_A
            u_hat_f_B[m, basis_function_index, :] = int_B
        
    sigma_hat_f_C_is = []
    u_hat_f_C_is = []

    for i in range(total_num_analyzed_rvs):
        C_i = np.zeros((mc_sample_size, total_num_rvs))
        if i < size_of_xi_e:
            for param_index in range(total_num_rvs):
                # Case E
                if param_index == i:
                    # print(f"Case E: i:{i}, size_of_xi_e:{size_of_xi_e}, param_index:{param_index}")
                    C_i[:, param_index] = A[:, param_index]
                else:
                    C_i[:, param_index] = B[:, param_index]
        elif i < size_of_xi_e + size_of_xi_v:
            # Case V
            for param_index in range(total_num_rvs):
                if i - size_of_xi_e == param_index - randomFieldE.J:
                    # print(f"Case V: i:{i}, size_of_xi_e:{size_of_xi_e}, param_index:{param_index}, randomFieldE.J:{randomFieldE.J}")
                    C_i[:, param_index] = A[:, param_index]
                else:
                    C_i[:, param_index] = B[:, param_index]
        else:
            # Case q
            for param_index in range(total_num_rvs):
                if param_index == randomFieldE.J + randomFieldVBar.J:
                    # print(f"Case V: i:{i}, size_of_xi_e:{size_of_xi_e}, param_index:{param_index}, randomFieldE.J:{randomFieldE.J}, randomFieldVBar.J:{randomFieldVBar.J}")
                    C_i[:, param_index] = A[:, param_index]
                else:
                    C_i[:, param_index] = B[:, param_index]
        
        sigma_hat_f_C_i = np.zeros((mc_sample_size, M))
        u_hat_f_C_i = np.zeros((mc_sample_size, N, 2))
        for m in range(mc_sample_size):
            print(f"Second loop xi {i+1} of {size_of_xi_e + size_of_xi_v + 1}, Iteration {m+1}/{mc_sample_size}")
            u_sol_C_i_data, sigma_hat_C_i_data = solve_model(fem_res, C_i[m, :randomFieldE.J], C_i[m, randomFieldE.J:randomFieldE.J + randomFieldVBar.J], C_i[m, randomFieldE.J + randomFieldVBar.J], randomFieldE, randomFieldVBar)
            u_sol_C_i = fe.Function(V)
            u_sol_C_i.vector()[:] = u_sol_C_i_data
            sigma_hat_f_C_i[m] = sigma_hat_C_i_data
            for basis_function_index, basis_function in enumerate(quadrature_basis_functions):
                transformation_matrix, transformation_vector = find_affine_transformation(basis_function.vertex_coords)
                active_quad_points = QUAD_POINTS_2DD_6
                quad_points = [Quad_point(np.dot(transformation_matrix, orig_quad_point.point) + transformation_vector, orig_quad_point.weight * 2 * basis_function.triangle_area) for orig_quad_point in active_quad_points]
                int_C_i = 0
                for quad_point in quad_points:
                    int_C_i += u_sol_C_i(quad_point.point) * quad_point.weight
                u_hat_f_C_i[m, basis_function_index, :] = int_C_i
        sigma_hat_f_C_is.append(sigma_hat_f_C_i)
        u_hat_f_C_is.append(u_hat_f_C_i)
    base_path = f"sobol_data_storage/klres_e_{kl_res_e}_klres_v_{kl_res_v}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}_size_of_xi_v_{size_of_xi_v}"
    os.makedirs(base_path, exist_ok=True)
    sigma_hat_f_A_path = os.path.join(base_path, 'sigma_hat_f_A.npy')
    sigma_hat_f_B_path = os.path.join(base_path, 'sigma_hat_f_B.npy')
    u_hat_f_A_path = os.path.join(base_path, 'u_hat_f_A.npy')
    u_hat_f_B_path = os.path.join(base_path, 'u_hat_f_B.npy')

    if os.path.exists(sigma_hat_f_A_path):
        sigma_hat_f_A_existing = np.load(sigma_hat_f_A_path)
        sigma_hat_f_A = np.concatenate((sigma_hat_f_A_existing, sigma_hat_f_A), axis=0)
    if os.path.exists(sigma_hat_f_B_path):
        sigma_hat_f_B_existing = np.load(sigma_hat_f_B_path)
        sigma_hat_f_B = np.concatenate((sigma_hat_f_B_existing, sigma_hat_f_B), axis=0)
    if os.path.exists(u_hat_f_A_path):
        u_hat_f_A_existing = np.load(u_hat_f_A_path)
        u_hat_f_A = np.concatenate((u_hat_f_A_existing, u_hat_f_A), axis=0)
    if os.path.exists(u_hat_f_B_path):
        u_hat_f_B_existing = np.load(u_hat_f_B_path)
        u_hat_f_B = np.concatenate((u_hat_f_B_existing, u_hat_f_B), axis=0)

    np.save(sigma_hat_f_A_path, sigma_hat_f_A)
    np.save(sigma_hat_f_B_path, sigma_hat_f_B)
    np.save(u_hat_f_A_path, u_hat_f_A)
    np.save(u_hat_f_B_path, u_hat_f_B)

    for i in range(total_num_analyzed_rvs):
        sigma_hat_f_C_i_path = os.path.join(base_path, f'sigma_hat_f_C_{i}.npy')
        u_hat_f_C_i_path = os.path.join(base_path, f'u_hat_f_C_{i}.npy')
        if os.path.exists(sigma_hat_f_C_i_path):
            sigma_hat_f_C_i_existing = np.load(sigma_hat_f_C_i_path)
            sigma_hat_f_C_is[i] = np.concatenate((sigma_hat_f_C_i_existing, sigma_hat_f_C_is[i]), axis=0)
        if os.path.exists(u_hat_f_C_i_path):
            u_hat_f_C_i_existing = np.load(u_hat_f_C_i_path)
            u_hat_f_C_is[i] = np.concatenate((u_hat_f_C_i_existing, u_hat_f_C_is[i]), axis=0)
        np.save(sigma_hat_f_C_i_path, sigma_hat_f_C_is[i])
        np.save(u_hat_f_C_i_path, u_hat_f_C_is[i])

    print("Sobol data saved")

def sp_6_sobol_calc_indices_u_hat(fem_res: int, kl_res_e: int, kl_res_v: int, size_of_xi_e: int, size_of_xi_v: int) -> tuple[np.array, np.array]:
    base_path = f"sobol_data_storage/klres_e_{kl_res_e}_klres_v_{kl_res_v}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}_size_of_xi_v_{size_of_xi_v}"
    
    total_num_analyzed_rvs = size_of_xi_e + size_of_xi_v + 1

    f_A = np.load(os.path.join(base_path, 'u_hat_f_A.npy')) # shape: (mc_sample_size, N, 2)
    f_B = np.load(os.path.join(base_path, 'u_hat_f_B.npy')) # shape: (mc_sample_size, N, 2)
    f_C_is = [np.load(os.path.join(base_path, f'u_hat_f_C_{i}.npy')) for i in range(total_num_analyzed_rvs)] # shape: (mc_sample_size, N, 2)

    S_single = np.zeros(total_num_analyzed_rvs)
    S_total = np.zeros(total_num_analyzed_rvs)

    basis_functions = get_quadrature_basis_functions(create_reference_mesh(fem_res))
    weights = np.array([basis_function.triangle_area for basis_function in basis_functions]).reshape(-1, 1)  # Reshape to (N, 1)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0) # shape (N, 2)

    for i in range(total_num_analyzed_rvs):
        f_C_i = f_C_is[i]

        var_g_q_i = (1 / f_A.shape[0]) * np.sum(f_A * f_C_i, axis=0) - f_0_squared # shape (N, 2)
        var_g_q_i_tilde = (1 / f_A.shape[0]) * np.sum(f_B * f_C_i, axis=0) - f_0_squared # shape (N, 2)
        var_g = (1 / f_A.shape[0]) * np.sum(f_A**2, axis=0) - f_0_squared # shape (N, 2)

        numerator_s_single = np.sum(weights * var_g_q_i) # shape (2)
        numerator_s_total = np.sum(weights * var_g_q_i_tilde) # shape (2)
        denominator = np.sum(weights * var_g) # shape (2)

        S_single[i] = np.sum(numerator_s_single) / np.sum(denominator)
        S_total[i] = 1 - np.sum(numerator_s_total) / np.sum(denominator)

    return S_single, S_total, f_A.shape[0]

def sp_6_sobol_calc_indices_sigma_hat(fem_res: int, kl_res_e: int, kl_res_v: int, size_of_xi_e: int, size_of_xi_v: int) -> tuple[np.array, np.array]:
    base_path = f"sobol_data_storage/klres_e_{kl_res_e}_klres_v_{kl_res_v}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}_size_of_xi_v_{size_of_xi_v}"

    total_num_analyzed_rvs = size_of_xi_e + size_of_xi_v + 1

    f_A_data = np.load(os.path.join(base_path, 'sigma_hat_f_A.npy'))
    f_B_data = np.load(os.path.join(base_path, 'sigma_hat_f_B.npy'))
    f_C_is_data = [np.load(os.path.join(base_path, f'sigma_hat_f_C_{i}.npy')) for i in range(total_num_analyzed_rvs)]

    mesh = create_reference_mesh(fem_res)
    W = fe.TensorFunctionSpace(mesh, 'P', 1)
    P_hat = fe.Point(0.16, 0.18)

    f_A = np.zeros((f_A_data.shape[0]))
    f_B = np.zeros((f_B_data.shape[0]))

    for i in range(f_A_data.shape[0]):
        sigma_hat_A = fe.Function(W)
        sigma_hat_A.vector()[:] = f_A_data[i]
        print(f"sigma_hat_A(P_hat): {sigma_hat_A(P_hat)}")
        f_A[i] = sigma_hat_A(P_hat)[0]

        sigma_hat_B = fe.Function(W)
        sigma_hat_B.vector()[:] = f_B_data[i]
        f_B[i] = sigma_hat_B(P_hat)[0]

    S_single = np.zeros(total_num_analyzed_rvs)
    S_total = np.zeros(total_num_analyzed_rvs)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    for i in range(total_num_analyzed_rvs):
        f_C_i = np.zeros((f_C_is_data[i].shape[0]))
        for j in range(f_C_is_data[i].shape[0]):
            sigma_hat_C = fe.Function(W)
            sigma_hat_C.vector()[:] = f_C_is_data[i][j]
            f_C_i[j] = sigma_hat_C(P_hat)[0]

        var_g_q_i = (1 / f_A.shape[0]) * np.sum(f_A * f_C_i, axis=0) - f_0_squared
        var_g_q_i_tilde = (1 / f_A.shape[0]) * np.sum(f_B * f_C_i, axis=0) - f_0_squared
        var_g = (1 / f_A.shape[0]) * np.sum(f_A**2, axis=0) - f_0_squared

        S_single[i] = var_g_q_i / var_g
        S_total[i] = 1 - var_g_q_i_tilde / var_g

    return S_single, S_total, f_A.shape[0]

def sp_6_plot_sobols(S_single: np.array, S_total: np.array, mc_sample_size: int, size_of_xi_e: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set width for each bar
    bar_width = 0.35

    ax.bar(np.arange(len(S_single)), S_single, width=bar_width, label='First Order')
    ax.bar(np.arange(len(S_single)) + bar_width, S_total, width=bar_width, label='Total Effect')
    x_labels_e = [fr"$\xi_{{{i+1}}}^{{E}}$" for i in range(size_of_xi_e)]
    x_labels_v = [fr"$\xi_{{{i+1}}}^{{V}}$" for i in range(len(S_single) - size_of_xi_e - 1)]
    x_labels = x_labels_e + x_labels_v + [r"$q$"]
    ax.set_xticklabels(x_labels, fontsize=24)
    ax.set_xticks(np.arange(len(S_single)) + bar_width / 2)
    ax.set_ylabel('Sensitivity [-]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=20)
    plt.show()
    print(f"Sample size: {mc_sample_size}")



# Image Creation
def create_reference_mesh(mesh_resolution: int) -> fe.Mesh:
    domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) # used a_plate_length hardcoded
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = 0.02 # used r hardcoded
    domain = domain - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    return mesh

def perturb_mesh(mesh: fe.Mesh, omega: np.array, r: float, randomFieldVBar: RandomFieldVBar) -> fe.Mesh:
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega, r, randomFieldVBar)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh
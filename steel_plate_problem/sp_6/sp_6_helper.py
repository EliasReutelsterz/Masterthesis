import fenics as fe
import numpy as np
from scipy.linalg import eig
import mshr
import os
import time
import sympy as sp
import matplotlib.pyplot as plt


#! LARGE FACTOR INVOLVED FOR NOW
alpha = 1E3
beta = 50


# General helpers

x1, x2, y1, y2 = sp.symbols('x1 x2 y1 y2')

v_cov1_1_sym = alpha * 5.0/100.0 * sp.exp(-4.0 * ((x1 - y1)**2 + (x2 - y2)**2))
v_cov1_2_sym = alpha * 1.0/100.0 * sp.exp(beta * -0.1 * ((2*x1 - y1)**2 + (2*x2 - y2)**2))
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
    return RandomFieldVBar(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J)
    

# Random field V

def perturbation_function(x: np.array, omega_2: np.array, r: float, randomFieldVBar: RandomFieldVBar) -> np.array:
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

    circ_perturb = np.array(randomFieldVBar(x_circ_proj, omega_2))

    x_pert = h / h_max * circ_perturb + (1 - h / h_max) * bound_perturb

    return np.array([0.16, 0.16]) + x_pert

def perturb_mesh(mesh: fe.Mesh, omega_2: np.array, r: float, randomFieldVBar: RandomFieldVBar) -> fe.Mesh:
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega_2, r, randomFieldVBar)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh

class RandomFieldVExpression(fe.UserExpression):
    def __init__(self, r: float, omega_2: np.array, domain: fe.Mesh, randomFieldVBar: RandomFieldVBar, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.omega_2 = omega_2
        self.domain = domain
        self.randomFieldVBar = randomFieldVBar

    def eval(self, values, x):
        perturbed_point = perturbation_function(x=x, omega_2=self.omega_2, r=self.r, randomFieldVBar=self.randomFieldVBar)
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
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, omega_2: np.array, r: float, randomFieldVBar: RandomFieldVBar, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi
        self.omega_2 = omega_2
        self.r = r
        self.randomFieldVBar = randomFieldVBar

    def eval(self, values, x):
        x_pert = perturbation_function(x=x, omega_2=self.omega_2, r=self.r, randomFieldVBar=self.randomFieldVBar)
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

def sample_omega_1(randomFieldE: RandomFieldE) -> np.array:
    return np.random.normal(0, 1, size=randomFieldE.J)

def sample_omega_2(randomFieldVBar: RandomFieldVBar) -> np.array:
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

def solve_model(mesh_resolution: int, omega_1: np.array, omega_2: np.array, q: float, randomFieldE: RandomFieldE = None, randomFieldVBar: RandomFieldVBar = None) -> fe.Function:

    # Model parameters
    a_plate_length = 0.32
    r = 0.02
    nu = 0.29
    rho = 7850
    grav = 6.6743*1E-11
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
    randomFieldVExpression = RandomFieldVExpression(omega_2=omega_2, r=r, domain=mesh, randomFieldVBar=randomFieldVBar)

    # Random field E \circ V
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=omega_1, omega_2=omega_2, r=r, randomFieldVBar=randomFieldVBar)


    deg = 1
    V = fe.VectorFunctionSpace(mesh, 'P', deg)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    u_hat_sol = fe.Function(V)

    # Boundary conditions
    ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    bc_left = fe.DirichletBC(V, fe.Constant((0,0)), left_boundary_function)
    g = fe.Constant((q,0))


    left_prefactor = fe.Expression('E / (2 * (1 + nu))', degree=1, E=randomFieldEExpression, nu=nu, domain=mesh)
    right_prefactor = fe.Expression('E / (2 * (1 - nu))', degree=1, E=randomFieldEExpression, nu=nu, domain=mesh)


    randomFieldEProj = fe.project(randomFieldVExpression, V)
    W = fe.TensorFunctionSpace(mesh, 'P', deg)
    jacobianProj = fe.project(fe.grad(randomFieldEProj), W)

    J_inv_T = J_minus_TExpression(jacobianProj, domain=mesh)
    J_helper1 = J_helper1Expression(jacobianProj, domain=mesh)
    J_helper2 = J_helper2Expression(jacobianProj, domain=mesh)
    det_J = J_determinantExpression(jacobianProj, domain=mesh)
    inv_det_J = J_inv_determinantExpression(jacobianProj, domain=mesh)

    left_integrand = det_J * fe.inner(fe.dot(J_inv_T, fe.grad(u)), fe.dot(J_inv_T, fe.grad(left_prefactor * v)))

    right_integrand = inv_det_J * \
                    (fe.dot(J_helper1, fe.grad(u)[:, 0]) + fe.dot(J_helper2, fe.grad(u)[:, 1])) * \
                    (fe.dot(J_helper1, fe.grad(right_prefactor * v)[:, 0]) + fe.dot(J_helper2, fe.grad(right_prefactor * v)[:, 1]))


    a = (left_integrand + right_integrand) * fe.dx
    # Right-hand side of weak form
    L = det_J * fe.inner(b,v) * fe.dx + det_J * fe.dot(g,v) * ds(3)

    # Solve Galerkin system
    fe.solve(a==L, u_hat_sol, bc_left)
    
    return u_hat_sol


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
        omega_1 = sample_omega_1(randomFieldE)
        omega_2 = sample_omega_2(randomFieldVBar)
        q = sample_q()
        u_hat_sols[i] = solve_model(mesh_resolution=mesh_resolution, omega_1=omega_1, omega_2=omega_2, q=q, randomFieldE=randomFieldE, randomFieldVBar=randomFieldVBar).vector().get_local().reshape((-1, 2))
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

#! Still to do!
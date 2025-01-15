import fenics as fe
import numpy as np
from scipy.linalg import eig
import mshr
import os
import time

# Random field V

def perturbation_function(x: np.array, omega: np.array, r: float) -> np.array:
    x = x - np.array([0.16, 0.16])
    c = np.sqrt(x[0]**2 + x[1]**2)
    x_circ_proj = r/c * x

    theta = np.arctan2(x[1], x[0]) # order has to be y, x as tan(y/x)=theta

    right_left = False
    if -np.pi/4 <= theta <= np.pi/4:
        x_bound_proj = np.array([0.16, 0.16*np.tan(theta)])
        right_left = True
    elif np.pi/4 <= theta <= 3*np.pi/4:
        x_bound_proj = np.array([0.16 / np.tan(theta), 0.16])
    elif theta <= -3*np.pi/4 or theta >= 3*np.pi/4:
        x_bound_proj = np.array([-0.16, -0.16*np.tan(theta)])
        right_left = True
    else:
        x_bound_proj = np.array([-0.16 / np.tan(theta), -0.16])

    h_max = np.sqrt((x_bound_proj[0] - x_circ_proj[0])**2 + (x_bound_proj[1] - x_circ_proj[1])**2)
    h = np.sqrt((x[0] - x_bound_proj[0])**2 + (x[1] - x_bound_proj[1])**2)

    bound_perturb = x_bound_proj

    circ_perturb = np.array([omega[0] * x_circ_proj[0] + omega[1], omega[0] * x_circ_proj[1] + omega[2]])

    x_pert = h / h_max * circ_perturb + (1 - h / h_max) * bound_perturb

    return np.array([0.16, 0.16]) + x_pert

def perturb_mesh(mesh: fe.Mesh, omega: np.array, r: float) -> fe.Mesh:
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega, r)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh

class RandomFieldVExpression(fe.UserExpression):
    def __init__(self, r: float, omega: np.array, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.omega = omega
        self.domain = domain

    def eval(self, values, x):
        perturbed_point = perturbation_function(x=x, omega=self.omega, r=self.r)
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
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, omega_2: np.array, r: float, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi
        self.omega_2 = omega_2
        self.r = r

    def eval(self, values, x):
        x_pert = perturbation_function(x=x, omega=self.omega_2, r=self.r)
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
 
def sample_omega_1(randomFieldE: RandomFieldE) -> np.array:
    return np.random.normal(0, 1, size=randomFieldE.J)

def sample_omega_2() -> np.array:
    return np.array([np.random.uniform(low=1/2, high=2),
                  np.random.uniform(low=-0.1, high=0.1),
                  np.random.uniform(low=-0.1, high=0.1)])

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

def solve_model(mesh_resolution: int, omega_1: np.array, omega_2: np.array, q: float, randomFieldE: RandomFieldE = None):

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
    randomFieldVExpression = RandomFieldVExpression(omega=omega_2, r=r, domain=mesh)

    # Random field E \circ V
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=omega_1, omega_2=omega_2, r=r)


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

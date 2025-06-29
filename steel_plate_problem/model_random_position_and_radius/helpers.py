import fenics as fe
import numpy as np
from scipy.linalg import eig
import mshr
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from IPython.display import clear_output
from matplotlib import patches


# Random field V

def perturbation_function(x: np.array, omega: np.array) -> np.array:
    """Perturbation function implementation.
    
    Args: 
        x: Coordinates of the point to be perturbed.
        omega: Perturbation parameters.
    Returns:
        Perturbed coordinates.
    """
    x = x - np.array([0.16, 0.16])
    c = np.sqrt(x[0]**2 + x[1]**2)
    x_circ_proj = 0.02/c * x

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

    circ_perturb = np.array([omega[0] * x_circ_proj[0] + omega[1], omega[0] * x_circ_proj[1] + omega[2]])

    x_pert = h / h_max * circ_perturb + (1 - h / h_max) * bound_perturb

    return np.array([0.16, 0.16]) + x_pert

def perturb_mesh(mesh: fe.Mesh, omega: np.array) -> fe.Mesh:
    """Perturb the whole reference mesh using the perturbation function.
    
    Args:
        mesh: Reference mesh to be perturbed.
        omega: Perturbation parameters.
    Returns:
        perturbed mesh.
    """
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh

class RandomFieldVExpression(fe.UserExpression):
    """Random field V expression for the FEM."""
    def __init__(self, omega: np.array, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega
        self.domain = domain

    def eval(self, values, x: np.array):
        perturbed_point = perturbation_function(x=x, omega=self.omega)
        values[0] = perturbed_point[0]
        values[1] = perturbed_point[1]

    def value_shape(self):
        return (2,)
    
class J_minus_TExpression(fe.UserExpression):
    """Transposed inverse of the projected Jacobian of the Random Field V."""
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x: np.array):
        J = self.jacobianProj(x).reshape((2, 2))
        J_inv_T = np.linalg.inv(J).T
        values[0] = J_inv_T[0, 0]
        values[1] = J_inv_T[0, 1]
        values[2] = J_inv_T[1, 0]
        values[3] = J_inv_T[1, 1]

    def value_shape(self):
        return (2, 2)
    
class J_helper1Expression(fe.UserExpression):
    """Helper expression for the Jacobian (J_22, - J_21)."""
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x: np.array):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = J[1, 1]
        values[1] = - J[1, 0]

    def value_shape(self):
        return (2, )

class J_helper2Expression(fe.UserExpression):
    """Helper expression for the Jacobian (- J_12, J_11)."""
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x: np.array):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = - J[0, 1]
        values[1] = J[0, 0]

    def value_shape(self):
        return (2, )
    
class J_helper3Expression(fe.UserExpression):
    """Helper expression for the Jacobian (sqrt(J_11^2 + J_12^2))."""
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x: np.array):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = np.sqrt(J[0, 0]**2 + J[0, 1]**2)

    def value_shape(self):
        return ()
    
class J_determinantExpression(fe.UserExpression):
    """Expression for the determinant of the Jacobian J."""
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x: np.array):
        J_det = np.linalg.det(self.jacobianProj(x).reshape((2, 2)))
        if J_det == 0:
            print("Determinant 0.")
            values[0] = 1
        else:
            values[0] = J_det

    def value_shape(self):
        return ()
    
class J_inv_determinantExpression(fe.UserExpression):
    """Expression for the inverse of the determinant of the Jacobian J."""
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x: np.array):
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
    """Class for quadrature points with weights."""
    def __init__(self, point: list, weight: float):
        self.point = point
        self.weight = weight

# Quadrature points for 2D triangle integration
QUAD_POINTS_2DD_6 = [Quad_point([(6 - np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(9 + 2 * np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(6 - np.sqrt(15)) / 21, (9 + 2 * np.sqrt(15)) / 21], (155 - np.sqrt(15)) / 2400),
                        Quad_point([(6 + np.sqrt(15)) / 21, (9 - 2 * np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([(6 + np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([(9 - 2 * np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21], (155 + np.sqrt(15)) / 2400),
                        Quad_point([1 / 3, 1 / 3], 9/80)]

def find_affine_transformation(triangle: np.array) -> tuple[np.array, np.array]:
    """Find the affine transformation matrix and vector for a triangle."""
    transformation_matrix = np.array([[triangle[1, 0] - triangle[0, 0], triangle[2, 0] - triangle[0, 0]],
                                    [triangle[1, 1] - triangle[0, 1], triangle[2, 1] - triangle[0, 1]]])
    transformation_vector = np.array([triangle[0, 0], triangle[0, 1]])
    return transformation_matrix, transformation_vector

def triangle_area(vertices: np.array):
    """Calculate the area of a triangle given its vertices."""
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    # which is the determinant of the transformation matrix
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

class ConstBasisFunction():
    """Class for constant basis functions with triangular support."""
    def __init__(self, basis_function: fe.Function, vertex_coords: np.array):
        self.function = basis_function
        self.vertex_coords = vertex_coords
        self.triangle_area = triangle_area(vertex_coords)
        self.middle_point = np.mean(vertex_coords, axis=0)

class RandomFieldE():
    """Class for the random field E."""
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array, basis_functions: list[ConstBasisFunction], N: int, J: int, mu: float):
        """Initialize the random field E.
        
        Args:
            eigenvalues: Eigenvalues to the covariance kernel.
            eigenvectors: Eigenvectors to the covariance kernel.
            basis_functions: Basis functions for the random field.
            N: Number of basis functions.
            J: KL truncation level.
            mu: Mean of the underlying gaussian random field.
        """
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J
        self.mu = mu

    def __call__(self, x: np.array, xi: np.array) -> float:
        index_supported_basis_function = self.find_supported_basis_function(x)
        return np.exp(self.mu + sum([np.sqrt(self.eigenvalues[m]) * self.eigenvectors[index_supported_basis_function, m] for m in range(len(xi))]))
    
    def find_supported_basis_function(self, x: np.array) -> int:
        """Find the index of the only supported basis function for the given x."""
        for i, basis_function in enumerate(self.basis_functions):
            if basis_function.function(x) == 1:
                return i
        raise ValueError("No supported basis function found for x")

class RandomFieldEExpression(fe.UserExpression):
    """Random field E expression for the FEM."""
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi

    def eval(self, values, x: np.array):
        values[0] = self.randomFieldE(x, self.xi)
    
    def value_shape(self):
        return ()

class RandomFieldEHatExpression(fe.UserExpression):
    """Random field Ê expression for the FEM with perturbation."""
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, omega2: np.array, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi
        self.omega2 = omega2

    def eval(self, values, x):
        x_pert = perturbation_function(x=x, omega=self.omega2)
        values[0] = self.randomFieldE(x_pert, self.xi)
    
    def value_shape(self):
        return ()
    
def matern_covariance_function(sigma: float, x: np.array, y: np.array) -> float:
    """Matern covariance function for 2D input and nu = 1/2."""
    # nu = 1/2
    l = 0.02
    h = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    return sigma**2 * np.exp(-h/l)

def get_C_entry_randomFieldE(sigma: float, basis_function_i: ConstBasisFunction, basis_function_j: ConstBasisFunction) -> float:
    """Calculate the covariance entry C_{ij} for the random field E.
    
    Args:
        sigma: Standard deviation of the matern covariance function.
        basis_function_i: Basis function i.
        basis_function_j: Basis function j.
    """
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
    """Calculate the random field E using the Karhunen-Loeve expansion.
    
    Args:
        mesh_resolution: Resolution of the mesh.
    Returns:
        RandomFieldE object with eigenvalues, eigenvectors, basis functions, N, J and mu.
    """
    
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
 
# Sample functions

def sample_omega1(randomFieldE: RandomFieldE) -> np.array:
    """Sample omega1 from a standard normal distribution.
    
    Args:
        randomFieldE: RandomFieldE object.
    Returns:
        Sample of omega1 of size randomFieldE.J.
    """
    return np.random.normal(0, 1, size=randomFieldE.J)

def sample_omega2() -> np.array:
    """Sample omega2 from a uniform distribution.
    
    Args:
        None
    Returns:
        Sample of omega2 of size 3.
    """
    return np.array([np.random.uniform(low=1/2, high=2),
                  np.random.uniform(low=-0.1, high=0.1),
                  np.random.uniform(low=-0.1, high=0.1)])

def sample_q() -> float:
    """Sample q from a normal distribution.

    Args:
        None
    Returns:
        Sample of q of size 1.
    """
    mu_q = 60*1E6
    sigma_q = 12*1E6
    return np.random.normal(mu_q, sigma_q)

def euclidean_distance(x: np.array, y: np.array) -> float:
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def create_reference_mesh(mesh_resolution: int) -> fe.Mesh:
    """Create the reference mesh for the domain."""
    domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) # used a_plate_length hardcoded
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = 0.02 # used r hardcoded
    domain = domain - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    return mesh

# Boundary classes and functions

class InnerCircleBoundary(fe.SubDomain):
    """Inner circle boundary class."""
    def __init__(self, mesh: fe.Mesh):
        super().__init__()  # Call the __init__ method of fe.SubDomain
        first_cell = fe.Cell(mesh, 0)
        self.vertex_length = first_cell.h()
        self.circ_center = fe.Point(0.16, 0.16)
        self.circ_radius = 0.02

    def inside(self, x: np.array, on_boundary: bool) -> bool:
        dist_to_center = euclidean_distance(x, self.circ_center)
        return on_boundary and dist_to_center < self.circ_radius + self.vertex_length/3

class LeftBoundary(fe.SubDomain):
    """Left boundary class."""
    def inside(self, x: np.array, on_boundary: bool) -> bool:
        return on_boundary and fe.near(x[0], 0)
        
def left_boundary_function(x, on_boundary: bool) -> bool:
    """Left boundary function."""
    return on_boundary and fe.near(x[0], 0)

class RightBoundary(fe.SubDomain):
    """Right boundary class."""
    def inside(self, x: np.array, on_boundary: bool) -> bool:
        return on_boundary and fe.near(x[0], 0.32)

def get_inner_circle_boundary_points(mesh: fe.Mesh) -> np.array:
    """Get the inner circle boundary points."""
    first_cell = fe.Cell(mesh, 0)
    vertex_length = first_cell.h()
    inner_circle_boundary_points = []
    for point in mesh.coordinates():
        if euclidean_distance(point, np.array([0.16, 0.16])) < 0.02 + vertex_length/3:
            inner_circle_boundary_points.append(point)
    return np.array(inner_circle_boundary_points)

def get_left_boundary_points(mesh: fe.Mesh) -> np.array:
    """Get the left boundary points."""
    left_boundary_points = []
    for point in mesh.coordinates():
        if fe.near(point[0], 0):
            left_boundary_points.append(point)
    return np.array(left_boundary_points)

def get_right_boundary_points(mesh: fe.Mesh) -> np.array:
    """Get the right boundary points."""
    right_boundary_points = []
    for point in mesh.coordinates():
        if fe.near(point[0], 0.32):
            right_boundary_points.append(point)
    return np.array(right_boundary_points)

# Main function

def solve_model(mesh_resolution: int, omega1: np.array, omega2: np.array, q: float, randomFieldE: RandomFieldE = None) -> np.array:
    """Solve the steel plate model using FEM for a given set of samples.
    
    Args:
        mesh_resolution: Resolution of the mesh.
        omega1: Sample for the random field E.
        omega2: Sample for the random field V.
        q: Sample for the load.
        randomFieldE (optional): RandomFieldE object.
    Returns:
        u_hat_sol_data: Solution vector for the displacement.
        sigma_hat_proj_data: Solution vector for the stress.
    """

    # Model parameters
    a_plate_length = 0.32
    r = 0.02
    nu = 0.29
    rho = 7850
    grav = 9.80665
    b = fe.Constant((0, -rho * grav))

    # Mesh
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = 0.02
    domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)

    # mark all vertices
    boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)

    # mark left boundary
    class LeftBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 0)
    left_boundary = LeftBoundary()
    left_boundary.mark(boundary_markers, 2)

    def left_boundary_function(x, on_boundary):
        return on_boundary and fe.near(x[0], 0)

    # mark right boundary
    class RightBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 0.32)
    right_boundary = RightBoundary()
    right_boundary.mark(boundary_markers, 3)


    # Random field V
    randomFieldVExpression = RandomFieldVExpression(omega=omega2, domain=mesh)

    # Random field Ê and E
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEHatExpression = RandomFieldEHatExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2)

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

    deg_test = 1
    V_test = fe.VectorFunctionSpace(mesh, 'P', deg_test)
    randomFieldVProj_test = fe.project(randomFieldVExpression, V_test)

    W_test = fe.TensorFunctionSpace(mesh, 'P', deg_test)
    jacobianProj = fe.project(fe.grad(randomFieldVProj_test), W_test)

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


    def sigma_hat(u_hat: fe.Function) -> fe.Expression:
        """Stress tensor defined on the reference domain."""
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

def solve_model_with_plots(mesh_resolution: int, omega1: np.array, omega2: np.array, q: float, randomFieldE: RandomFieldE = None) -> None:
    """Solve the steel plate model using FEM for a given set of samples and plot the results.
    This function exists mainly for testing purposes.
    
    Args:
        mesh_resolution: Resolution of the mesh.
        omega1: Sample for the random field E.
        omega2: Sample for the random field V.
        q: Sample for the load.
        randomFieldE (optional): RandomFieldE object.
    Returns:
        None
    """

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
    perturbed_mesh = perturb_mesh(mesh=mesh, omega=omega2)
    randomFieldVExpression = RandomFieldVExpression(omega=omega2, domain=mesh)

    # Random field Ê and E
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEHatExpression = RandomFieldEHatExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2)
    randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=omega1)


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
        perturbed_inner_circle_boundary_points.append(perturbation_function(x=point, omega=omega2))
    perturbed_inner_circle_boundary_points = np.array(perturbed_inner_circle_boundary_points)

    # Plot perturbed mesh
    plt.subplot(2, 4, 5)
    fe.plot(perturbed_mesh, title=r'Perturbed Mesh for $\Omega(\omega_2)$')
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
    c = fe.plot(E_proj, title=r"Random Field $E(V(\hat{x}, \omega_2), \omega_1)$")
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
    c = fe.plot(E_perturbed_proj, title=r"Random Field $E(x, \omega_1)$")
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

def save_mc_samples(u_hat_sols: np.array, sigma_hat_proj_data: np.array, mesh_resolution_kl_e: int, mesh_resolution: int) -> None:
    """Save the Monte Carlo samples to a file.
    
    Args:
        u_hat_sols: Array of u_hat solutions.
        sigma_hat_proj_data: Array of sigma_hat_proj data.
        mesh_resolution_kl_e: Mesh resolution for KL expansion.
        mesh_resolution: Mesh resolution for FEM.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    base_path = f'mc_data_storage/klres_e_{mesh_resolution_kl_e}_femres_{mesh_resolution}'
    os.makedirs(base_path, exist_ok=True)
    # u_hat
    file_path_u_hat = os.path.join(base_path, 'u_hat_sols.npy')
    if os.path.exists(file_path_u_hat):
        u_hat_sols_existing = np.load(file_path_u_hat)
        u_hat_sols = np.concatenate((u_hat_sols_existing, u_hat_sols), axis=0)
    np.save(file_path_u_hat, u_hat_sols)
    # sigma_hat_proj
    file_path_sigma_hat_proj = os.path.join(base_path, 'sigma_hat_proj.npy')
    if os.path.exists(file_path_sigma_hat_proj):
        sigma_hat_proj_existing = np.load(file_path_sigma_hat_proj)
        sigma_hat_proj_data = np.concatenate((sigma_hat_proj_existing, sigma_hat_proj_data), axis=0)
    np.save(file_path_sigma_hat_proj, sigma_hat_proj_data)
    print(f"Samples saved!")

def run_and_save_mc(mesh_resolution_kl_e: int, mesh_resolution: int, sample_size: int, randomFieldE: RandomFieldE = None) -> None:
    """Run the Monte Carlo simulation and save the results.
    
    Args:
        mesh_resolution_kl_e: Mesh resolution for KL expansion.
        mesh_resolution: Mesh resolution for FEM.
        sample_size: Number of Monte Carlo samples.
        randomFieldE (optional): RandomFieldE object.
    Returns:
        None
    """
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution_kl_e)
    mesh = create_reference_mesh(mesh_resolution=mesh_resolution)
    u_hat_sols = np.zeros((sample_size, mesh.coordinates().shape[0] * 2))
    sigma_hat_proj_datas = []
    for i in range(sample_size):
        print(f"Iteration {i + 1} of {sample_size}")
        omega1 = sample_omega1(randomFieldE)
        omega2 = sample_omega2()
        q = sample_q()
        u_hat_sols[i], sigma_hat_proj_data = solve_model(mesh_resolution=mesh_resolution, omega1=omega1, omega2=omega2, q=q, randomFieldE=randomFieldE)
        sigma_hat_proj_datas.append(sigma_hat_proj_data)
    save_mc_samples(u_hat_sols, np.array(sigma_hat_proj_datas), mesh_resolution_kl_e, mesh_resolution)

def mc_analysis_u_hat(mesh_resolution_kl_e: int, mesh_resolution: int) -> None:
    """Perform Monte Carlo analysis on the u_hat solutions.
    
    Args:
        mesh_resolution_kl_e: Mesh resolution for KL expansion.
        mesh_resolution: Mesh resolution for FEM.
    Returns:
        None
    """
    
    data_u_hat_sols = np.load(f'mc_data_storage/klres_e_{mesh_resolution_kl_e}_femres_{mesh_resolution}/u_hat_sols.npy')

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [32]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= data_u_hat_sols.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------

    if data_u_hat_sols.shape[0] - np.sum(mc_sample_sizes) < mc_sample_sizes[-1]:
        mc_sample_sizes = mc_sample_sizes[:-1]

    fine_u_hat_sols = data_u_hat_sols[np.sum(mc_sample_sizes):]
    sparse_u_hat_sols = data_u_hat_sols[:np.sum(mc_sample_sizes)]

    print(f"fine_u_hat_sols.shape: {fine_u_hat_sols.shape}")
    print(f"mc_sample_sizes: {mc_sample_sizes}")


    # Mean and variance of fine solutions
    mesh = create_reference_mesh(mesh_resolution)
    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    fine_mean_u_hat_sols = fe.Function(V)

    fine_mean_u_hat_sols_array = np.mean(fine_u_hat_sols, axis=0)
    fine_mean_u_hat_sols.vector()[:] = fine_mean_u_hat_sols_array
    
    # Plot fine solution
    x_coords = mesh.coordinates()[:, 0]
    y_coords = mesh.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[0:0.32:500j, 0:0.32:500j]
    
    circle = plt.Circle((0.16, 0.16), 0.02, color='black', fill=False)

    fig_mean, ax = plt.subplots(figsize=(10, 8))
    cp = fe.plot(fine_mean_u_hat_sols)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    ax.set_title(r'Mean estimation $\hat{u}(\hat{x},\omega)$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tight_layout()
    plt.show()

    # Calculate 2x2 Covariance matrix
    U = fe.FunctionSpace(mesh, 'P', 1)
    fine_var_u_hat_sols_11 = fe.Function(U)
    fine_var_u_hat_sols_12 = fe.Function(U)
    fine_var_u_hat_sols_21 = fe.Function(U)
    fine_var_u_hat_sols_22 = fe.Function(U)
    dof_coords = U.tabulate_dof_coordinates()
    n = fine_u_hat_sols.shape[0]
    for i, u_hat_sol_data in enumerate(fine_u_hat_sols):
        u_hat_sol = fe.Function(V)
        u_hat_sol.vector()[:] = u_hat_sol_data
        for j, point in enumerate(dof_coords):
            fine_var_u_hat_sols_11.vector()[j] += 1/(n - 1) * (u_hat_sol(point)[0] - fine_mean_u_hat_sols(point)[0])**2
            fine_var_u_hat_sols_12.vector()[j] += 1/(n - 1) * (u_hat_sol(point)[0] - fine_mean_u_hat_sols(point)[0])*(u_hat_sol(point)[1] - fine_mean_u_hat_sols(point)[1])
            fine_var_u_hat_sols_21.vector()[j] += 1/(n - 1) * (u_hat_sol(point)[1] - fine_mean_u_hat_sols(point)[1])*(u_hat_sol(point)[0] - fine_mean_u_hat_sols(point)[0])
            fine_var_u_hat_sols_22.vector()[j] += 1/(n - 1) * (u_hat_sol(point)[1] - fine_mean_u_hat_sols(point)[1])**2

    # Plot variances
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    variance_indices = [11, 12, 21, 22]

    for var_comp_index, var_function in enumerate([fine_var_u_hat_sols_11, fine_var_u_hat_sols_12, fine_var_u_hat_sols_21, fine_var_u_hat_sols_22]):
        ax = axs.flat[var_comp_index]
        z_values_fine_var = []
        for i in range(len(x_coords)):
            z_values_fine_var.append(var_function(x_coords[i], y_coords[i]))
        grid_z_var = griddata((x_coords, y_coords), z_values_fine_var, (grid_x, grid_y), method='linear')
        mask_grid_pert = (grid_x - 0.16)**2 + (grid_y - 0.16)**2 <= 0.02**2
        grid_z_var = np.ma.masked_where(mask_grid_pert, grid_z_var)  # Use masked array

        cp = ax.contourf(grid_x, grid_y, grid_z_var, levels=100, cmap='viridis')
        cbar = plt.colorbar(cp)
        cbar.ax.tick_params(labelsize=20)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        # cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        cbar.ax.yaxis.get_offset_text().set_fontsize(20)
        ax.set_title(rf'$Cov_{{{variance_indices[var_comp_index]}}}$', fontsize=24)
        if var_comp_index == 0 or var_comp_index == 2:
            ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
        if var_comp_index == 2 or var_comp_index == 3:
            ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)        
        ax.set_xlim(-0.02, 0.34)
        ax.set_ylim(-0.02, 0.34)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.tight_layout()
    plt.show()

    # Convergence analysis

    sparse_u_hat_sols_functions = []
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
        
        # Calculate L2 and H1 errors
        L2_errors.append(fe.errornorm(fine_mean_u_hat_sols, sparse_mean_sol, 'L2'))
        H1_errors.append(fe.errornorm(fine_mean_u_hat_sols, sparse_mean_sol, 'H1'))

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

def mc_analysis_sigma_hat(mesh_resolution_kl_e: int, mesh_resolution: int, P_hat: fe.Point) -> None:
    """Perform Monte Carlo analysis on the sigma_hat solutions.
    
    Args:
        mesh_resolution_kl_e: Mesh resolution for KL expansion.
        mesh_resolution: Mesh resolution for FEM.
        P_hat: Evaluation Point in the reference domain.
    Returns:
        None
    """

    data_sigma_hat_proj = np.load(f"mc_data_storage/klres_e_{mesh_resolution_kl_e}_femres_{mesh_resolution}/sigma_hat_proj.npy")

    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------
    mc_sample_sizes = [32]
    while True:
        if np.sum(mc_sample_sizes) + mc_sample_sizes[-1]*2 <= data_sigma_hat_proj.shape[0]:
            mc_sample_sizes.append(mc_sample_sizes[-1]*2)
        else:
            break
    # ----------- THIS PART CAN BE CHANGED FOR DIFFERENT MC SAMPLE SIZES ------------

    if data_sigma_hat_proj.shape[0] - np.sum(mc_sample_sizes) < mc_sample_sizes[-1]:
        mc_sample_sizes = mc_sample_sizes[:-1]

    fine_sigma_hat_proj = data_sigma_hat_proj[np.sum(mc_sample_sizes):]
    sparse_sigma_hat_proj = data_sigma_hat_proj[:np.sum(mc_sample_sizes)]

    print(f"fine_sigma_hat_proj.shape: {fine_sigma_hat_proj.shape}")
    print(f"mc_sample_sizes: {mc_sample_sizes}")

    # Mean and variance of fine solutions
    mesh = create_reference_mesh(mesh_resolution)
    W = fe.TensorFunctionSpace(mesh, 'P', 1)
    fine_mean_sigma_hat_proj = fe.Function(W)
    fine_mean_sigma_hat_proj_array = np.mean(fine_sigma_hat_proj, axis=0)
    fine_mean_sigma_hat_proj.vector()[:] = fine_mean_sigma_hat_proj_array

    fine_sigma_hat_proj_P_hat_var = 0
    for data_fine_single in fine_sigma_hat_proj:
        fine_sol = fe.Function(W)
        fine_sol.vector()[:] = data_fine_single
        fine_sigma_hat_proj_P_hat_var += (fine_sol(P_hat)[0] - fine_mean_sigma_hat_proj(P_hat)[0])**2
    fine_sigma_hat_proj_P_hat_var /= fine_sigma_hat_proj.shape[0] - 1
    upper_confidence_bound_fine = fine_mean_sigma_hat_proj(P_hat)[0] + 1.96 * np.sqrt(fine_sigma_hat_proj_P_hat_var / fine_sigma_hat_proj.shape[0])
    lower_confidence_bound_fine = fine_mean_sigma_hat_proj(P_hat)[0] - 1.96 * np.sqrt(fine_sigma_hat_proj_P_hat_var / fine_sigma_hat_proj.shape[0])

    # Plot Mean solution
    # mean: vectorized

    circle = plt.Circle((0.16, 0.16), 0.02, color='black', fill=False)

    fig_mean, ax = plt.subplots(figsize=(10, 8))
    cp = fe.plot(fine_mean_sigma_hat_proj[:, 0]) # Plot only first column
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(20)
    ax.set_title(r'Mean estimation $\hat{\sigma}(\hat{x}, \hat{u}, \omega) \cdot e_1$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.scatter(P_hat.x(), P_hat.y(), color='red', s=100, label=r'$\hat{P}$')
    ax.legend(loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.show()

    # mean: first component
    x_coords = mesh.coordinates()[:, 0]
    y_coords = mesh.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[0:0.32:500j, 0:0.32:500j]
    
    fig_mean, ax = plt.subplots(figsize=(10, 8))

    z_values_fine_mean = []
    for i in range(len(x_coords)):
        z_values_fine_mean.append(fine_mean_sigma_hat_proj(x_coords[i], y_coords[i])[0])
    grid_z_mean = griddata((x_coords, y_coords), z_values_fine_mean, (grid_x, grid_y), method='linear')
    mask_grid_pert = (grid_x - 0.16)**2 + (grid_y - 0.16)**2 <= 0.02**2
    grid_z_mean = np.ma.masked_where(mask_grid_pert, grid_z_mean)  # Use masked array

    cp = ax.contourf(grid_x, grid_y, grid_z_mean, levels=100, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(20)
    ax.set_title(r'Mean estimation $\hat{\sigma}_{11}(\hat{x}, \hat{u}, \omega)$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.scatter(P_hat.x(), P_hat.y(), color='red', s=100, label=r'$\hat{P}$')
    ax.legend(loc='upper left', fontsize=20)

    plt.tight_layout()
    plt.show()

    # Convergence Analysis at P_hat
    sparse_sigma_hat_proj_P_hat_means = []

    for mc_sample_index, mc_sample_size in enumerate(mc_sample_sizes):
        sparse_mean_sol = fe.Function(W)
        sparse_mean_sol.set_allow_extrapolation(True)

        # Calculate mean solution
        if mc_sample_index == 0:
            data_sparse_sample = sparse_sigma_hat_proj[:mc_sample_size]
        else:
            data_sparse_sample = sparse_sigma_hat_proj[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]
        sparse_mean_sol.vector()[:] = np.mean(data_sparse_sample, axis=0).reshape(-1)
        sparse_sigma_hat_proj_P_hat_means.append(sparse_mean_sol(P_hat)[0])

    # Calculate variance
    sparse_sigma_hat_proj_P_hat_vars = np.zeros(len(mc_sample_sizes))
    upper_confidence_bounds = []
    lower_confidence_bounds = []
    for mc_sample_index, mc_sample_size in enumerate(mc_sample_sizes):
        # Calculate mean solution
        if mc_sample_index == 0:
            data_sparse_sample = sparse_sigma_hat_proj[:mc_sample_size]
        else:
            data_sparse_sample = sparse_sigma_hat_proj[int(np.sum(mc_sample_sizes[:(mc_sample_index - 1)])):int(np.sum(mc_sample_sizes[:mc_sample_index]))]
        for data_sparse_single in data_sparse_sample:
            sparse_sol = fe.Function(W)
            sparse_sol.vector()[:] = data_sparse_single
            sparse_sigma_hat_proj_P_hat_vars[mc_sample_index] += (sparse_sol(P_hat)[0] - sparse_sigma_hat_proj_P_hat_means[mc_sample_index])**2
        sparse_sigma_hat_proj_P_hat_vars[mc_sample_index] /= mc_sample_size - 1
        upper_confidence_bounds.append(sparse_sigma_hat_proj_P_hat_means[mc_sample_index] + 1.96 * np.sqrt(sparse_sigma_hat_proj_P_hat_vars[mc_sample_index] / mc_sample_size))
        lower_confidence_bounds.append(sparse_sigma_hat_proj_P_hat_means[mc_sample_index] - 1.96 * np.sqrt(sparse_sigma_hat_proj_P_hat_vars[mc_sample_index] / mc_sample_size))

    # Plot convergence of mean solution at P_hat
    
    fig_var, ax = plt.subplots(figsize=(12, 6))
    # Code for plotting linear scale
    # ax.plot(mc_sample_sizes, sparse_sigma_hat_proj_P_hat_means, 'bo', marker='x', linestyle='None', label='Means', markersize=10)
    # ax.fill_between(mc_sample_sizes, lower_confidence_bounds, upper_confidence_bounds, alpha=0.2, label='95% Confidence Interval')
    # ax.axhline(y=fine_mean_sigma_hat_proj(P_hat)[0], color='r', linestyle='-', label='Mean reference solution')
    # ax.fill_between(mc_sample_sizes, lower_confidence_bound_fine, upper_confidence_bound_fine, alpha=0.2, color='red', label='95% Confidence Interval reference solution')
    # ax.set_ylabel('Mean', fontsize=24)
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Code for plotting log scale
    ax.plot(mc_sample_sizes, np.abs(sparse_sigma_hat_proj_P_hat_means - fine_mean_sigma_hat_proj(P_hat)[0]), 'bo', marker='x', linestyle='None', label='Absolute Error', markersize=10)
    max_abs_confidence_bounds = []
    for i in range(len(mc_sample_sizes)):
        max_abs_confidence_bounds.append(np.max([np.abs(lower_confidence_bounds[i] - fine_mean_sigma_hat_proj(P_hat)[0]), np.abs(upper_confidence_bounds[i] - fine_mean_sigma_hat_proj(P_hat)[0])]))
    ax.fill_between(mc_sample_sizes, max_abs_confidence_bounds, alpha=0.2, label='95% Confidence Interval')
    ax.set_yscale('log')
    ax.set_ylabel('Error to reference', fontsize=24)

    ax.set_xscale('log')
    ax.set_xlabel('MC Samples', fontsize=24)
    ax.legend(loc='upper left', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.get_offset_text().set_fontsize(20)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Sobol indices

def get_quadrature_basis_functions(mesh: fe.Mesh) -> list[ConstBasisFunction]:
    """Get piecewise constant quadrature basis functions with triangular support for the given mesh.
    
    Args:
        mesh: The mesh to create the basis functions for.
    Returns:
        A list of ConstBasisFunction objects representing the basis functions.
    """
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

def sp_5_sobol_calc_indices_u_hat(fem_res: int, kl_res_e:int, size_of_xi_e: int) -> tuple[np.array, np.array]:
    """Calculate Sobol indices for the u_hat solutions.
    
    Args:
        fem_res: Mesh resolution for FEM.
        kl_res_e: Mesh resolution for KL expansion.
        size_of_xi_e: Number of random variables of random field E to consider in the Sobol index calculation.
    Returns:
        A tuple containing the first-order Sobol indices and total Sobol indices.
    """

    base_path = f"sobol_data_storage/klres_e_{kl_res_e}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}"
    f_A = np.load(os.path.join(base_path, 'u_hat_f_A.npy')) # shape: (mc_sample_size, N, 2)
    f_B = np.load(os.path.join(base_path, 'u_hat_f_B.npy')) # shape: (mc_sample_size, N, 2)
    f_C_is = [np.load(os.path.join(base_path, f'u_hat_f_C_{i}.npy')) for i in range(size_of_xi_e + 4)] # shape: (mc_sample_size, N, 2)

    S_single = np.zeros(size_of_xi_e + 4)
    S_total = np.zeros(size_of_xi_e + 4)

    basis_functions = get_quadrature_basis_functions(create_reference_mesh(fem_res))
    weights = np.array([basis_function.triangle_area for basis_function in basis_functions]).reshape(-1, 1)  # Reshape to (N, 1)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0) # shape (N, 2)

    for i in range(size_of_xi_e + 4):
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

def sp_5_plot_sobols(S_single: np.array, S_total: np.array, mc_sample_size: int, size_of_xi_e: int) -> None:
    """Plot the Sobol indices.
    
    Args:
        S_single: First-order Sobol indices.
        S_total: Total Sobol indices.
        mc_sample_size: Number of Monte Carlo samples.
        size_of_xi_e: Number of random variables of random field E to consider in the Sobol index calculation.
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set width for each bar
    bar_width = 0.35

    ax.bar(np.arange(len(S_single)), S_single, width=bar_width, label='First Order')
    ax.bar(np.arange(len(S_single)) + bar_width, S_total, width=bar_width, label='Total Effect')
    x_labels = [fr"$\xi_{{{i+1}}}^{{E}}$" for i in range(size_of_xi_e)]
    x_labels.extend([r"$\omega_2^{(1)}$", r"$\omega_2^{(2)}$", r"$\omega_2^{(3)}$", r"$q$"])
    ax.set_xticklabels(x_labels, fontsize=24)
    ax.set_xticks(np.arange(len(S_single)) + bar_width / 2)
    ax.set_ylabel('Sensitivity [-]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize=20)
    plt.show()
    print(f"Sample size: {mc_sample_size}")

def sp_5_sobol_run_samples_and_save(mc_sample_size: int, fem_res: int, kl_res_e: int, size_of_xi_e: int, randomFieldE: RandomFieldE = None) -> None:
    """Run the Sobol sampling and save the results.
    
    Args:
        mc_sample_size: Number of Monte Carlo samples.
        fem_res: Mesh resolution for FEM.
        kl_res_e: Mesh resolution for KL expansion.
        size_of_xi_e: Number of random variables of random field E to consider in the Sobol index calculation.
        randomFieldE (optional): RandomFieldE object.
    Returns:
        None
    """
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(kl_res_e)
    
    mesh = create_reference_mesh(fem_res)
    quadrature_basis_functions = get_quadrature_basis_functions(mesh)
    N = len(quadrature_basis_functions)

    # total xis are: omega1, omega2, q (but only part of omega1 is investigated)
    len_xi_total = randomFieldE.J + 3 + 1
    A = np.zeros((mc_sample_size, len_xi_total))
    B = np.zeros((mc_sample_size, len_xi_total))

    A[:, :randomFieldE.J] = np.array([sample_omega1(randomFieldE) for _ in range(mc_sample_size)])
    B[:, :randomFieldE.J] = np.array([sample_omega1(randomFieldE) for _ in range(mc_sample_size)])
    A[:, randomFieldE.J:randomFieldE.J + 3] = np.array([sample_omega2() for _ in range(mc_sample_size)])
    B[:, randomFieldE.J:randomFieldE.J + 3] = np.array([sample_omega2() for _ in range(mc_sample_size)])
    A[:, randomFieldE.J + 3] = np.array([sample_q() for _ in range(mc_sample_size)])
    B[:, randomFieldE.J + 3] = np.array([sample_q() for _ in range(mc_sample_size)])

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
        u_sol_A_data, sigma_hat_A_data = solve_model(fem_res, A[m, :randomFieldE.J], A[m, randomFieldE.J:randomFieldE.J + 3], A[m, randomFieldE.J + 3], randomFieldE)
        u_sol_A = fe.Function(V)
        u_sol_A.vector()[:] = u_sol_A_data
        sigma_hat_f_A[m] = sigma_hat_A_data
        u_sol_B_data, sigma_hat_B_data = solve_model(fem_res, B[m, :randomFieldE.J], B[m, randomFieldE.J:randomFieldE.J + 3], B[m, randomFieldE.J + 3], randomFieldE)
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

    for i in range(size_of_xi_e + 4):
        C_i = np.zeros((mc_sample_size, len_xi_total))
        if i < size_of_xi_e:
            for param_index in range(len_xi_total):
                if param_index == i:
                    C_i[:, param_index] = A[:, param_index]
                else:
                    C_i[:, param_index] = B[:, param_index]
        else:
            for param_index in range(len_xi_total):
                if i - size_of_xi_e == param_index - randomFieldE.J:
                    C_i[:, param_index] = A[:, param_index]
                else:
                    C_i[:, param_index] = B[:, param_index]
        
        sigma_hat_f_C_i = np.zeros((mc_sample_size, M))
        u_hat_f_C_i = np.zeros((mc_sample_size, N, 2))
        for m in range(mc_sample_size):
            print(f"Second loop xi {i+1} of {size_of_xi_e + 4}, Iteration {m+1}/{mc_sample_size}")
            u_sol_C_i_data, sigma_hat_C_i_data = solve_model(fem_res, C_i[m, :randomFieldE.J], C_i[m, randomFieldE.J:randomFieldE.J + 3], C_i[m, randomFieldE.J + 3], randomFieldE)
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
    base_path = f'sobol_data_storage/klres_e_{kl_res_e}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}'
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

    for i in range(size_of_xi_e + 4):
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

def sp_5_sobol_calc_indices_sigma_hat(fem_res: int, kl_res_e:int, size_of_xi_e: int) -> tuple[np.array, np.array]:
    """Calculate Sobol indices for the sigma_hat solutions.
    
    Args:
        fem_res: Mesh resolution for FEM.
        kl_res_e: Mesh resolution for KL expansion.
        size_of_xi_e: Number of random variables of random field E to consider in the Sobol index calculation.
    Returns:
        A tuple containing the first-order Sobol indices and total Sobol indices.
    """
    base_path = f"sobol_data_storage/klres_e_{kl_res_e}_femres_{fem_res}_size_of_xi_e_{size_of_xi_e}"
    f_A_data = np.load(os.path.join(base_path, 'sigma_hat_f_A.npy'))
    f_B_data = np.load(os.path.join(base_path, 'sigma_hat_f_B.npy'))
    f_C_is_data = [np.load(os.path.join(base_path, f'sigma_hat_f_C_{i}.npy')) for i in range(size_of_xi_e + 4)]

    mesh = create_reference_mesh(fem_res)
    W = fe.TensorFunctionSpace(mesh, 'P', 1)
    P_hat = fe.Point(0.16, 0.18)

    f_A = np.zeros((f_A_data.shape[0]))
    f_B = np.zeros((f_B_data.shape[0]))

    for i in range(f_A_data.shape[0]):
        sigma_hat_A = fe.Function(W)
        sigma_hat_A.vector()[:] = f_A_data[i]
        f_A[i] = sigma_hat_A(P_hat)[0]

        sigma_hat_B = fe.Function(W)
        sigma_hat_B.vector()[:] = f_B_data[i]
        f_B[i] = sigma_hat_B(P_hat)[0]

    S_single = np.zeros(size_of_xi_e + 4)
    S_total = np.zeros(size_of_xi_e + 4)

    f_0_squared = np.mean(f_A, axis=0) * np.mean(f_B, axis=0)

    for i in range(size_of_xi_e + 4):
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


# Image creation helpers

def perturbation_function_with_more_returns(x: np.array, omega: np.array) -> tuple[np.array, np.array, np.array, np.array]:
    """Calculate perturbation function returning all of the projected points used in the image creation.
    
    Args:
        x: Input coordinates.
        omega: Random variables.
    Returns:
        A tuple containing the perturbed coordinates, circular projection, circular perturbation, and boundary perturbation.
    """
    x = x - np.array([0.16, 0.16])
    c = np.sqrt(x[0]**2 + x[1]**2)
    x_circ_proj = 0.02/c * x

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

    circ_perturb = np.array([omega[0] * x_circ_proj[0] + omega[1], omega[0] * x_circ_proj[1] + omega[2]])

    x_pert = h / h_max * circ_perturb + (1 - h / h_max) * bound_perturb

    return  np.array([0.16, 0.16]) + x_pert, \
            np.array([0.16, 0.16]) + x_circ_proj, \
            np.array([0.16, 0.16]) + circ_perturb, \
            np.array([0.16, 0.16]) + bound_perturb

def plot_reference_domain():
    """Plot the reference domain."""
    mesh_resolution = 32
    mesh = create_reference_mesh(mesh_resolution)
    inner_circle_boundary_points = get_inner_circle_boundary_points(mesh)
    left_boundary_points = get_left_boundary_points(mesh)
    right_boundary_points = get_right_boundary_points(mesh)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    fe.plot(mesh)
    plt.scatter(left_boundary_points[:, 0], left_boundary_points[:, 1], color='green', s=40, label=r'$\Gamma_1$')
    plt.scatter(right_boundary_points[:, 0], right_boundary_points[:, 1], color='blue', s=40, label=r'$\Gamma_2$')
    plt.quiver(right_boundary_points[:, 0], right_boundary_points[:, 1], np.ones_like(right_boundary_points[:, 0]), np.zeros_like(right_boundary_points[:, 1]), color='blue', scale=16)
    plt.scatter(inner_circle_boundary_points[:, 0], inner_circle_boundary_points[:, 1], color='cyan', s=40, label='Inner circle boundary')
    plt.scatter(0.16, 0.18, color='red', s=40, label='QoI point')
    plt.xlabel(r'$x_1$ [m]', fontsize=24)
    plt.ylabel(r'$x_2$ [m]', fontsize=24)
    plt.legend(loc='upper right', fontsize=20)
    plt.xlim(- 0.02, 0.35)
    plt.ylim(- 0.02, 0.35)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.show()

def plot_perturbed_mesh_sample():
    """Plot a perturbed mesh sample and the perturbation function."""

    mesh_resolution = 16
    mesh = create_reference_mesh(mesh_resolution)
    omega2 = np.array([1.5, 0.05, 0.05])
    perturbed_mesh = perturb_mesh(mesh, omega2)

    # Create points on the circle
    thetas = np.linspace(0, 2 * np.pi, 500)
    circle_points = np.array([0.16 + 0.02 * np.cos(thetas), 0.16 + 0.02 * np.sin(thetas)]).T
    perturbed_circle_points = np.array([perturbation_function(circle_point, omega2) for circle_point in circle_points])


    # Plots
    plt.figure(figsize=(24, 8))

    # Plot circle and perturbed points
    ax = plt.subplot(1, 3, 1)
    plt.scatter(circle_points[:, 0], circle_points[:, 1], label='Reference Circle', s=1)
    plt.scatter(perturbed_circle_points[:, 0], perturbed_circle_points[:, 1], label='Perturbed Circle', s=1)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    plt.xlabel(r'$x_1$ [m]', fontsize=30)
    plt.ylabel(r'$x_2$ [m]', fontsize=30)
    plt.legend(loc='upper right', fontsize=30, markerscale=10)
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    ax = plt.subplot(1, 3, 2)
    fe.plot(mesh, label=r'Reference Mesh $D_{ref}$')
    plt.xlabel(r'$\hat{x}_1$ [m]', fontsize=30)
    plt.ylabel(r'$\hat{x}_2$ [m]', fontsize=30)
    plt.legend(loc='upper right', fontsize=30)
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    ax = plt.subplot(1, 3, 3)
    fe.plot(perturbed_mesh, label=r'Perturbed Mesh $D(\omega_2)$')
    plt.xlabel(r'$x_1$ [m]', fontsize=30)
    plt.ylabel(r'$x_2$ [m]', fontsize=30)
    plt.legend(loc='upper right', fontsize=30)
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    plt.tight_layout()
    plt.show()

def plot_perturbation_function():
    """Plot the perturbation function."""

    omega2 = np.array([1.5, 0.05, 0.05])
    # Create points on the circle
    thetas = np.linspace(0, 2 * np.pi, 500)
    circle_points = np.array([0.16 + 0.02 * np.cos(thetas), 0.16 + 0.02 * np.sin(thetas)]).T
    perturbed_circle_points = np.array([perturbation_function(circle_point, omega2) for circle_point in circle_points])



    x = np.array([0.27, 0.21])
    x_pert, x_circ, x_circ_pert, x_bound = perturbation_function_with_more_returns(x, omega2)

    # Plot circle and perturbed points
    plt.figure(figsize=(16, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title(r'Reference domain $D_{ref}$', fontsize=30)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey', zorder=1)
    plt.scatter(circle_points[:, 0], circle_points[:, 1], label='Reference Circle', s=1, zorder=1)
    plt.scatter(x[0], x[1], label='Original Point', s=200, marker='x', color='cyan', linewidths=3, zorder=2)
    plt.scatter(x_circ[0], x_circ[1], label='Circle Projection', s=200, marker='x', linewidths=3, zorder=2)
    plt.scatter(x_bound[0], x_bound[1], label='Bound Projection', s=200, marker='x', linewidths=3, zorder=2)
    ax.plot([x[0], x_bound[0]], [x[1], x_bound[1]], color='grey', label=r'$h$', linestyle='dotted', zorder=1)
    ax.plot([x[0], x_circ[0]], [x[1], x_circ[1]], color='black', label=r'$h_{max} - h$', linestyle=(0, (5, 5)), zorder=1)
    plt.xlabel(r'$\hat{x}_1$ [m]', fontsize=30)
    plt.ylabel(r'$\hat{x}_2$ [m]', fontsize=30)
    lgnd = ax.legend(loc='lower left', fontsize=20)
    lgnd.legend_handles[0]._sizes = [30]
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    ax = plt.subplot(1, 2, 2)
    ax.set_title(r'Sample domain $D(\omega_2)$', fontsize=30)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey', zorder=1)
    plt.scatter(perturbed_circle_points[:, 0], perturbed_circle_points[:, 1], label='Perturbed Circle', s=1, zorder=1)
    plt.scatter(x_pert[0], x_pert[1], label='Perturbed Point', s=200, marker='x', color='cyan', linewidths=3, zorder=2)
    plt.scatter(x_circ_pert[0], x_circ_pert[1], label='Perturbed Circle Projection', s=200, marker='x', linewidths=3, zorder=2)
    plt.scatter(x_bound[0], x_bound[1], label='Bound Projection', s=200, marker='x', linewidths=3, zorder=2)
    plt.xlabel(r'$x_1$ [m]', fontsize=30)
    plt.ylabel(r'$x_2$ [m]', fontsize=30)
    lgnd = plt.legend(loc='lower left', fontsize=20)
    lgnd.legend_handles[0]._sizes = [30]
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    plt.tight_layout()
    plt.show()

def plot_random_field_e():
    """Plot the random field E and its perturbed version."""

    mesh_resolution = 16
    mesh = create_reference_mesh(mesh_resolution)
    randomFieldE = calculate_randomFieldE(mesh_resolution)

    omega1 = sample_omega1(randomFieldE)
    omega2 = np.array([1.5, 0.05, 0.05])
    randomFieldEHatExpression = RandomFieldEHatExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2)
    randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=omega1)
    x_coords = mesh.coordinates()[:, 0]
    y_coords = mesh.coordinates()[:, 1]
    grid_x, grid_y = np.mgrid[0:0.32:500j, 0:0.32:500j]


    # Ê
    z_values_E_hat = []
    for i in range(len(x_coords)):
        z_values_E_hat.append(randomFieldEHatExpression([x_coords[i], y_coords[i]]))

    grid_z_E_hat = griddata((x_coords, y_coords), z_values_E_hat, (grid_x, grid_y), method='linear')
    fig = plt.figure(figsize=(18, 8))
    ax = plt.subplot(1, 2, 1)

    center_x_pert = 0.16
    center_y_pert = 0.16
    radius_pert = 0.02

    mask_grid_pert = (grid_x - center_x_pert)**2 + (grid_y - center_y_pert)**2 <= radius_pert**2
    grid_z_E_hat_masked = np.ma.masked_where(mask_grid_pert, grid_z_E_hat)  # Use masked array

    cp = ax.contourf(grid_x, grid_y, grid_z_E_hat_masked, levels=100, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$\hat{E}(\hat{x}, \omega_1, \omega_2) \text{ on } D_{ref}$', fontsize=24)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tick_params(axis='both', which='major', labelsize=16)


    # E
    center_x = 0.21
    center_y = 0.21
    radius = 0.03

    z_values_E = []
    for i in range(len(x_coords)):
        z_values_E.append(randomFieldEExpression([x_coords[i], y_coords[i]]))

    grid_z_E = griddata((x_coords, y_coords), z_values_E, (grid_x, grid_y), method='linear')

    ax = plt.subplot(1, 2, 2)

    mask_grid = (grid_x - center_x)**2 + (grid_y - center_y)**2 <= radius**2
    grid_z_E_masked = np.ma.masked_where(mask_grid, grid_z_E)  # Use masked array

    cp = ax.contourf(grid_x, grid_y, grid_z_E_masked, levels=100, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlabel(r'$x_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$x_2$ [m]', fontsize=24)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$E(x, \omega_1) \text{ on } D(\omega_2)$', fontsize=24)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.show()

def plot_grid_perturbation_for_introduction():
    """Plot the perturbed grid for the introduction section."""

    mesh_resolution = 16
    mesh = create_reference_mesh(mesh_resolution)
    omega2 = np.array([1.5, 0.05, 0.05])
    perturbed_mesh = perturb_mesh(mesh, omega2)

    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # First plot
    ax1 = axs[0]
    plt.subplot(1, 2, 1)
    fe.plot(mesh, label=r'Reference Mesh $D_{ref}$', color='blue', linewidth=0.5)
    plt.xlabel(r'$\hat{x}_1$ [m]', fontsize=30)
    plt.ylabel(r'$\hat{x}_2$ [m]', fontsize=30)
    plt.legend(loc='upper right', fontsize=30)
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Second plot
    ax2 = axs[1]
    plt.subplot(1, 2, 2)
    fe.plot(perturbed_mesh, label=r'Perturbed Mesh $D(\omega_2)$', color='red', linewidth=0.5)
    plt.xlabel(r'$x_1$ [m]', fontsize=30)
    plt.ylabel(r'$x_2$ [m]', fontsize=30)
    plt.legend(loc='upper right', fontsize=30)
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Add first arrow
    xyA = [0.33, 0.25]
    xyB = [-0.01, 0.25]

    arrow = patches.ConnectionPatch(
        xyA,
        xyB,
        coordsA=ax1.transData,
        coordsB=ax2.transData,
        color="black",
        arrowstyle="-|>",
        mutation_scale=30,
        linewidth=3,
        connectionstyle="arc3,rad=-0.2"
    )
    fig.patches.append(arrow)

    # Label first arrow
    fig.text(0.515, 0.85, r'$V$', ha='center', va='center', fontsize=24)

    # Add second arrow
    xyA = [0.33, 0.07]
    xyB = [-0.01, 0.07]

    arrow = patches.ConnectionPatch(
        xyB,
        xyA,
        coordsA=ax2.transData,
        coordsB=ax1.transData,
        color="black",
        arrowstyle="-|>",
        mutation_scale=30,
        linewidth=3,
        connectionstyle="arc3,rad=-0.2"
    )
    fig.patches.append(arrow)

    # Label second arrow
    fig.text(0.52, 0.35, r'$V^{-1}$', ha='center', va='center', fontsize=24)

    plt.tight_layout()
    plt.show()

def plot_solution_sample_without_perturbation():
    """Plot the solution sample without perturbation."""

    mesh_resolution = 16
    randomFieldE = calculate_randomFieldE(mesh_resolution)
    omega1 = sample_omega1(randomFieldE)
    omega2 = np.array([1, 0, 0])
    q = sample_q()

    u_hat_data, sigma_hat_data = solve_model(mesh_resolution, omega1, omega2, q, randomFieldE)

    mesh = create_reference_mesh(mesh_resolution)
    perturbed_mesh = perturb_mesh(mesh, omega2)

    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    V_sigma = fe.TensorFunctionSpace(mesh, 'P', 1)
    V_pert = fe.VectorFunctionSpace(perturbed_mesh, 'P', 1)
    V_sigma_pert = fe.TensorFunctionSpace(perturbed_mesh, 'P', 1)

    u_hat = fe.Function(V)
    u_hat.vector()[:] = u_hat_data
    u = fe.Function(V_pert)
    u.vector()[:] = u_hat_data
    sigma_hat = fe.Function(V_sigma)
    sigma_hat.vector()[:] = sigma_hat_data
    sigma = fe.Function(V_sigma_pert)
    sigma.vector()[:] = sigma_hat_data

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # u
    ax = plt.subplot(1, 2, 1)
    circle = plt.Circle((0.16, 0.16), 0.02, color='black', fill=False)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    c = fe.plot(u)
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$u(x, \omega)$ on $D(\omega_2)$', fontsize=24)
    ax.set_xlabel(r'$x_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$x_2$ [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # sigma
    ax = plt.subplot(1, 2, 2)
    circle = plt.Circle((0.16, 0.16), 0.02, color='black', fill=False)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    c = fe.plot(sigma[:, 0])
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$\sigma(x, u, \omega) \cdot e_1$ on $D(\omega_2)$', fontsize=24)
    ax.set_xlabel(r'$x_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$x_2$ [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    plt.tight_layout()
    plt.show()

def plot_solution_sample_with_perturbation():
    """Plot the solution sample with perturbation."""

    mesh_resolution = 16
    randomFieldE = calculate_randomFieldE(mesh_resolution)
    omega1 = sample_omega1(randomFieldE)
    omega2 = np.array([1.5, 0.05, 0.05])
    q = sample_q()

    u_hat_data, sigma_hat_data = solve_model(mesh_resolution, omega1, omega2, q, randomFieldE)

    mesh = create_reference_mesh(mesh_resolution)
    perturbed_mesh = perturb_mesh(mesh, omega2)

    V = fe.VectorFunctionSpace(mesh, 'P', 1)
    V_sigma = fe.TensorFunctionSpace(mesh, 'P', 1)
    V_pert = fe.VectorFunctionSpace(perturbed_mesh, 'P', 1)
    V_sigma_pert = fe.TensorFunctionSpace(perturbed_mesh, 'P', 1)

    u_hat = fe.Function(V)
    u_hat.vector()[:] = u_hat_data
    u = fe.Function(V_pert)
    u.vector()[:] = u_hat_data
    sigma_hat = fe.Function(V_sigma)
    sigma_hat.vector()[:] = sigma_hat_data
    sigma = fe.Function(V_sigma_pert)
    sigma.vector()[:] = sigma_hat_data

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # u_hat
    ax = plt.subplot(2, 2, 1)
    circle = plt.Circle((0.16, 0.16), 0.02, color='black', fill=False)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    c = fe.plot(u_hat)
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$\hat{u}(\hat{x}, \omega)$ on $D_{ref}$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # u
    ax = plt.subplot(2, 2, 2)
    circle = plt.Circle((0.21, 0.21), 0.03, color='black', fill=False)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    c = fe.plot(u)
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$u(x, \omega)$ on $D(\omega_2)$', fontsize=24)
    ax.set_xlabel(r'$x_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$x_2$ [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # sigma_hat
    ax = plt.subplot(2, 2, 3)
    circle = plt.Circle((0.16, 0.16), 0.02, color='black', fill=False)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    c = fe.plot(sigma_hat[:, 0])
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$\hat{\sigma}(\hat{x}, \hat{u}, \omega) \cdot e_1$ on $D_{ref}$', fontsize=24)
    ax.set_xlabel(r'$\hat{x}_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$\hat{x}_2$ [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # sigma
    ax = plt.subplot(2, 2, 4)
    circle = plt.Circle((0.21, 0.21), 0.03, color='black', fill=False)
    ax.add_artist(circle)
    ax.plot([0, 0.32, 0.32, 0, 0], [0, 0, 0.32, 0.32, 0], color='lightgrey')
    c = fe.plot(sigma[:, 0])
    cbar = plt.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(-0.02, 0.34)
    ax.set_ylim(-0.02, 0.34)
    ax.set_title(r'$\sigma(x, u, \omega) \cdot e_1$ on $D(\omega_2)$', fontsize=24)
    ax.set_xlabel(r'$x_1$ [m]', fontsize=24)
    ax.set_ylabel(r'$x_2$ [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    plt.tight_layout()
    plt.show()
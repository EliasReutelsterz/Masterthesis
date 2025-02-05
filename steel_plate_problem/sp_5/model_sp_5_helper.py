import fenics as fe
import numpy as np
from scipy.linalg import eig
import mshr
import os
import time
import matplotlib.pyplot as plt


# Random field V

def perturbation_function(x: np.array, omega: np.array) -> np.array:
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
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh

class RandomFieldVExpression(fe.UserExpression):
    def __init__(self, omega: np.array, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega
        self.domain = domain

    def eval(self, values, x):
        perturbed_point = perturbation_function(x=x, omega=self.omega)
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
        values[1] = J[0, 0]

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
    def __init__(self, randomFieldE: RandomFieldE, xi: np.array, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldE = randomFieldE
        self.xi = xi

    def eval(self, values, x):
        values[0] = self.randomFieldE(x, self.xi)
    
    def value_shape(self):
        return ()

class RandomFieldEHatExpression(fe.UserExpression):
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
 
# Sample functions

def sample_omega1(randomFieldE: RandomFieldE) -> np.array:
    return np.random.normal(0, 1, size=randomFieldE.J)

def sample_omega2() -> np.array:
    return np.array([np.random.uniform(low=1/2, high=2),
                  np.random.uniform(low=-0.1, high=0.1),
                  np.random.uniform(low=-0.1, high=0.1)])

def sample_q() -> float:
    mu_q = 60*1E6
    sigma_q = 12*1E6
    return np.random.normal(mu_q, sigma_q)

def euclidean_distance(x, y):
        return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def create_reference_mesh(mesh_resolution: int) -> fe.Mesh:
    domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) # used a_plate_length hardcoded
    circ_center = fe.Point(0.16, 0.16)
    circ_radius = 0.02 # used r hardcoded
    domain = domain - mshr.Circle(circ_center, circ_radius)
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    return mesh

# Boundary classes and functions

class InnerCircleBoundary(fe.SubDomain):
    def __init__(self, mesh: fe.Mesh):
        super().__init__()  # Call the __init__ method of fe.SubDomain
        first_cell = fe.Cell(mesh, 0)
        self.vertex_length = first_cell.h()
        self.circ_center = fe.Point(0.16, 0.16)
        self.circ_radius = 0.02

    def inside(self, x, on_boundary):
        dist_to_center = euclidean_distance(x, self.circ_center)
        return on_boundary and dist_to_center < self.circ_radius + self.vertex_length/2

class LeftBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 0)
        
def left_boundary_function(x, on_boundary):
        return on_boundary and fe.near(x[0], 0)

class RightBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], 0.32)

def get_inner_circle_boundary_points(mesh: fe.Mesh):
    first_cell = fe.Cell(mesh, 0)
    vertex_length = first_cell.h()
    inner_circle_boundary_points = []
    for point in mesh.coordinates():
        if euclidean_distance(point, np.array([0.16, 0.16])) < 0.02 + vertex_length/2:
            inner_circle_boundary_points.append(point)
    return np.array(inner_circle_boundary_points)

def get_left_boundary_points(mesh: fe.Mesh):
    left_boundary_points = []
    for point in mesh.coordinates():
        if fe.near(point[0], 0):
            left_boundary_points.append(point)
    return np.array(left_boundary_points)

def get_right_boundary_points(mesh: fe.Mesh):
    right_boundary_points = []
    for point in mesh.coordinates():
        if fe.near(point[0], 0.32):
            right_boundary_points.append(point)
    return np.array(right_boundary_points)

def solve_and_plot_model(mesh_resolution: int, omega1: np.array, omega2: np.array, q: float, randomFieldE: RandomFieldE = None, plot: bool = False):

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

    # Random field E \circ V #! \circ V is important
    if not randomFieldE:
        randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
    randomFieldEHatExpression = RandomFieldEHatExpression(randomFieldE=randomFieldE, xi=omega1, omega2=omega2)
    randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=omega1)


    deg = 1
    V = fe.VectorFunctionSpace(mesh, 'P', deg)
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
    jacobianProj = fe.project(fe.grad(randomFieldVProj_test), W_test) #! consider higher degree polynomials

    J_inv_T = J_minus_TExpression(jacobianProj, domain=mesh)
    J_helper1 = J_helper1Expression(jacobianProj, domain=mesh)
    J_helper2 = J_helper2Expression(jacobianProj, domain=mesh)
    det_J = J_determinantExpression(jacobianProj, domain=mesh)
    inv_det_J = J_inv_determinantExpression(jacobianProj, domain=mesh)

    left_integrand = det_J * fe.inner(fe.dot(J_inv_T, fe.grad(u_hat)), fe.dot(J_inv_T, fe.grad(left_prefactor * v_hat)))

    G_hat = fe.as_matrix([[fe.dot(J_helper1, fe.grad(u_hat)[:, 0]) + (2 * nu)/(1 + nu) * fe.dot(J_helper2, fe.grad(u_hat)[:, 1]),
                        (1 - nu)/(1 + nu) * fe.dot(J_helper2, fe.grad(u_hat)[:, 0])],
                        [(1 - nu)/(1 + nu) * fe.dot(J_helper1, fe.grad(u_hat)[:, 1]),
                            (2 * nu)/(1 + nu) * fe.dot(J_helper1, fe.grad(u_hat)[:, 0]) + fe.dot(J_helper2, fe.grad(u_hat)[:, 1])]])

    right_integrand = fe.inner(G_hat, fe.dot(J_inv_T, fe.grad(right_prefactor * v_hat)))


    a = (left_integrand + right_integrand) * fe.dx
    # Right-hand side of weak form
    L = det_J * fe.dot(b,v_hat) * fe.dx + fe.dot(g,v_hat) * ds(3)

    # Solve Galerkin system
    fe.solve(a==L, u_hat_sol, bc_left)

    if not plot:
        return u_hat_sol

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
    V_perturbed = fe.VectorFunctionSpace(perturbed_mesh, "P", deg)
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

    # Plot Sigma(û)
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
        return E_hat * inv_det_J * fe.as_matrix([[entry_11, entry_12], [entry_12, entry_22]])

    sigma_hat_proj = fe.project(sigma_hat(u_hat_sol)[:, 0], V_test)
    for right_bounday_point in right_boundary_points:
        print(f"sigma_hat_proj({right_bounday_point}): {sigma_hat_proj(right_bounday_point)}")

    plt.subplot(2, 4, 3)
    c = fe.plot(sigma_hat_proj, title=r'$\sigma(\hat{u}) \cdot e_1$')
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
    W = fe.FunctionSpace(mesh, 'P', deg)
    E_proj = fe.project(randomFieldEHatExpression, W)
    plt.subplot(2, 4, 4)
    c = fe.plot(E_proj, title=r"Random Field $E(V(\hat{x}, \omega_2), \omega_1)$")
    plt.colorbar(c)
    plt.xlabel(r'$\hat{x}_1$')
    plt.ylabel(r'$\hat{x}_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    # Plot random field E(x)
    W_perturbed = fe.FunctionSpace(perturbed_mesh, "P", deg)
    E_perturbed_proj = fe.Function(W_perturbed)
    E_perturbed_proj.vector()[:] = E_proj.vector()[:]
    plt.subplot(2, 4, 8)
    c = fe.plot(E_perturbed_proj, title=r"Random Field $E(x, \omega_1)$")
    plt.colorbar(c)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(- 0.02, 0.34)
    plt.ylim(- 0.02, 0.34)

    plt.tight_layout()
    plt.show()

    return u_hat_sol
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
        self.geometric_dimension = 2  # Specify the geometric dimension

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
    

def solve_model(mesh_resolution: int = 12):
    # Material parameters
    E = np.exp(26.011)
    q = 60 * 1e6
    g = fe.Constant((q, 0))
    rho = 7850
    g_gravity = 9.80665
    b = fe.Constant((0, - rho * g_gravity))
    nu = 0.29

    # Random field
    omega2 = np.array([1, 0, 0])
    #! omega2 = np.array([1.5, 0.05, 0.05])

    # Mesh
    domain_ref = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) - mshr.Circle(fe.Point(0.16, 0.16), 0.02)
    mesh_ref = mshr.generate_mesh(domain_ref, mesh_resolution)
    mesh_ref = fe.Mesh(mesh_ref)
    
    # Varational formulation
    V = fe.VectorFunctionSpace(mesh_ref, "P", 1)

    randomFieldVExpression = RandomFieldVExpression(omega=omega2, domain=mesh_ref, degree=1)
    #! randomFieldVProj = fe.project(randomFieldVExpression, V)
    # define randomFieldVProj as identity
    randomFieldVProj = fe.Function(V)
    randomFieldVProj.interpolate(fe.Expression(("x[0]", "x[1]"), degree=1))

    #! jacobianProj = fe.project(fe.grad(randomFieldVProj), fe.TensorFunctionSpace(mesh_ref, "P", 1))
    # define jacobianproj as identity matrix
    jacobianProj = fe.Function(fe.TensorFunctionSpace(mesh_ref, "P", 1))
    jacobianProj.interpolate(fe.Expression((("1", "0"), ("0", "1")), degree=1))
    j_minus_T = J_minus_TExpression(jacobianProj=jacobianProj, domain=mesh_ref, degree=1)
    j_helper1 = J_helper1Expression(jacobianProj=jacobianProj, domain=mesh_ref, degree=1)
    j_helper2 = J_helper2Expression(jacobianProj=jacobianProj, domain=mesh_ref, degree=1)


    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # Mark boundaries
    class RightBoundary(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and np.isclose(x[0], 0.32)
            
    class LeftBoundary(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and np.isclose(x[0], 0)

    boundary_markers = fe.MeshFunction("size_t", mesh_ref, mesh_ref.topology().dim()-1, 0)
    boundary_markers.set_all(9999)
    right_boundary = RightBoundary()
    right_boundary.mark(boundary_markers, 2)
    left_boundary = LeftBoundary()
    left_boundary.mark(boundary_markers, 1)
    ds = fe.Measure("ds", domain=mesh_ref, subdomain_data=boundary_markers)
    # Left boundary
    
    bc_gamma1 = fe.DirichletBC(V, fe.Constant((0, 0)), boundary_markers, 1)
    
    

    a_left = fe.det(jacobianProj) * E/(2*(1+nu)) * fe.inner(fe.dot(j_minus_T, fe.grad(u)), fe.dot(j_minus_T, fe.grad(v))) * fe.dx
    G = fe.as_matrix([[fe.dot(j_helper1, fe.grad(u)[:, 0]) + (2*nu)/(1+nu) * fe.dot(j_helper2, fe.grad(u)[:, 1]),
                     (1-nu)/(1+nu) * fe.dot(j_helper2, fe.grad(u)[:, 0])],
                     [(1-nu)/(1+nu) * fe.dot(j_helper1, fe.grad(u)[:, 1]),
                      (2*nu)/(1+nu)*fe.dot(j_helper1, fe.grad(u)[:, 0]) + fe.dot(j_helper2, fe.grad(u)[:, 1])]])
    a_right = E/(2*(1-nu)) * fe.inner(G, fe.dot(j_minus_T, fe.grad(v))) * fe.dx



    a =  a_left + a_right
    l = fe.det(jacobianProj) * fe.inner(b, v) * fe.dx + fe.det(jacobianProj) * fe.inner(g, v) * ds(2) #! consider adding det J here

    def sigma(u):
        u_grad = fe.grad(u)
        return fe.as_matrix([[E/(1-nu**2) * (u_grad[0, 0] + nu * u_grad[1, 1]),
                              E/(2*(1+nu)) * (u_grad[0, 1] + u_grad[1, 0])],
                             [E/(2*(1+nu)) * (u_grad[0, 1] + u_grad[1, 0]),
                              E/(1-nu**2) * (u_grad[1, 1] + nu * u_grad[0, 0])]])
    W = fe.TensorFunctionSpace(mesh_ref, "P", 1)



    # Solve
    u_sol = fe.Function(V)
    fe.solve(a == l, u_sol, bc_gamma1)

    # Plot solution
    fe.plot(u_sol)
    plt.show()

    sigma_proj = fe.project(sigma(u_sol)[:, 0], V)

    right_boundary_test_points = [fe.Point(0.32, y) for y in np.linspace(0, 0.32, 10)]
    for right_bounday_point in right_boundary_test_points:
        print(f"sigma_proj({right_bounday_point}): {sigma_proj(right_bounday_point)}")

    fe.plot(sigma_proj)
    plt.show()

    fe.plot(randomFieldVProj, title="Random field V")
    plt.show()

    c = fe.plot(jacobianProj[:, 0], title="First column Jacobian of V")
    plt.colorbar(c)
    plt.show()

    c = fe.plot(jacobianProj[:, 1], title="Second column Jacobian of V")
    plt.colorbar(c)
    plt.show()

if __name__ == "__main__":
    solve_model()
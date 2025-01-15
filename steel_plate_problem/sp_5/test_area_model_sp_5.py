from __future__ import print_function
import fenics as fe
import mshr
import numpy as np
import matplotlib.pyplot as plt
import os

from model_sp_5_helper import *

current_dir = os.path.dirname(os.path.abspath(__file__))


# Design parameters
mesh_resolution = 12

# Model parameters
a_plate_length = 0.32
r = 0.02
nu = 0.29
rho = 7850
grav = 6.6743*1E-11
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

# Save mesh
mesh_path = os.path.join(current_dir, 'data_output/sp_4_reference_mesh.xml')
fe.File(mesh_path) << mesh

# Random q
mu_q = 60*1E6
sigma_q = 12*1E6
q = np.random.normal(mu_q, sigma_q)

# Random field V
omega_2 = np.array([np.random.uniform(low=1/2, high=2),
                  np.random.uniform(low=-0.1, high=0.1),
                  np.random.uniform(low=-0.1, high=0.1)])
perturbed_mesh = perturb_mesh(mesh=mesh, omega=omega_2, r=r)
randomFieldVExpression = RandomFieldVExpression(omega=omega_2, r=r, domain=mesh)

# Random field E \circ V #! \circ V is important
randomFieldE = calculate_randomFieldE(mesh_resolution=mesh_resolution)
omega_1 = np.random.normal(0, 1, size=randomFieldE.J)
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


randomFieldVProj = fe.project(randomFieldVExpression, V)
W = fe.TensorFunctionSpace(mesh, 'P', deg)
jacobianProj = fe.project(fe.grad(randomFieldVProj), W)

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
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Mark inner circle boundary points for perturbed mesh
perturbed_inner_circle_boundary_points = []
for point in inner_circle_boundary_points:
    perturbed_inner_circle_boundary_points.append(perturbation_function(x=point, omega=omega_2, r=r))
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
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot the solution û
plt.subplot(2, 4, 2)
c = fe.plot(u_hat_sol, title=r'Displacement $\hat{u}(\hat{x}, \omega)$')
plt.colorbar(c)
plt.title(r'Displacement $\hat{u}(\hat{x}, \omega)$')
plt.xlabel(r'$\hat{x}_1$')
plt.ylabel(r'$\hat{x}_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot the solution u
V_perturbed = fe.VectorFunctionSpace(perturbed_mesh, "P", deg)
u_sol = fe.Function(V_perturbed)
u_sol.vector()[:] = u_hat_sol.vector()[:] 
plt.subplot(2, 4, 6)
c = fe.plot(u_sol, title=r'Displacement $u(x, \omega)$')
plt.colorbar(c)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot Sigma(û)
def sigma(u): # Stress tensor
    return det_J * left_prefactor * fe.dot(J_inv_T, fe.grad(u)) + right_prefactor * (fe.dot(J_helper1, fe.grad(u)[:, 0]) + fe.dot(J_helper2, fe.grad(u)[:, 1])) * fe.Identity(2)

def sigma_div_det_J(u):
    return sigma(u) / det_J

sigma_proj = fe.project(sigma_div_det_J(u_hat_sol)[:, 0], V)
plt.subplot(2, 4, 3)
c = fe.plot(sigma_proj, title=r'$\sigma(\hat{u}) \cdot e_1$')
plt.colorbar(c)
plt.xlabel(r'$\hat{x}_1$')
plt.ylabel(r'$\hat{x}_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot Sigma(u)
sigma_perturbed_proj = fe.Function(V_perturbed)
sigma_perturbed_proj.vector()[:] = sigma_proj.vector()[:]
plt.subplot(2, 4, 7)
c = fe.plot(sigma_perturbed_proj, title=r'$\sigma(u) \cdot e_1$')
plt.colorbar(c)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot random field E(V(^x))
W = fe.FunctionSpace(mesh, 'P', deg)
E_proj = fe.project(randomFieldEExpression, W)
plt.subplot(2, 4, 4)
c = fe.plot(E_proj, title=r"Random Field $E(V(\hat{x}, \omega_2), \omega_1)$")
plt.colorbar(c)
plt.xlabel(r'$\hat{x}_1$')
plt.ylabel(r'$\hat{x}_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot random field E(x)
W_perturbed = fe.FunctionSpace(perturbed_mesh, "P", deg)
E_perturbed_proj = fe.Function(W_perturbed)
E_perturbed_proj.vector()[:] = E_proj.vector()[:]
plt.subplot(2, 4, 8)
c = fe.plot(E_perturbed_proj, title=r"Random Field $E(x, \omega_1)$")
plt.colorbar(c)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

plt.tight_layout()
plt.show()
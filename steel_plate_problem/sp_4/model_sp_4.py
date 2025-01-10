from __future__ import print_function
import fenics as fe
import mshr
import numpy as np
import matplotlib.pyplot as plt
import os

from model_sp_4_helper import *

current_dir = os.path.dirname(os.path.abspath(__file__))


# Design parameters
mesh_resolution = 32

# Model parameters
a_plate_length = 0.32
r = 0.02
nu = 0.29
rho = 7850
grav = 6.6743*1E-11
b = fe.Constant((0, -rho * grav))
q = 60*1E6
E = 2*1E11


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
def euclidean_distance(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
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


def plot_mesh_with_linewidth(mesh, ax, linewidth=0.2):
    coordinates = mesh.coordinates()
    cells = mesh.cells()
    for cell in cells:
        polygon = coordinates[cell]
        polygon = np.append(polygon, [polygon[0]], axis=0)  # Close the polygon
        ax.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=linewidth)

# Save mesh
mesh_path = os.path.join(current_dir, 'data_output/sp_4_reference_mesh.xml')
fe.File(mesh_path) << mesh

omega = np.array([np.random.uniform(low=1/2, high=2),
                  np.random.uniform(low=-0.1, high=0.1),
                  np.random.uniform(low=-0.1, high=0.1)])
perturbed_mesh = perturb_mesh(mesh=mesh, omega=omega, r=r)


deg = 1
V = fe.VectorFunctionSpace(mesh, 'P', deg)
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
u_hat_sol = fe.Function(V)

# Boundary conditions
ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
bc_left = fe.DirichletBC(V, fe.Constant((0,0)), left_boundary_function)

g = fe.Constant((q,0))


left_prefactor = fe.Expression('E / (2 * (1 + nu))', degree=1, E=E, nu=nu, domain=mesh)
right_prefactor = fe.Expression('E / (2 * (1 - nu))', degree=1, E=E, nu=nu, domain=mesh)

randomFieldVExpression = RandomFieldVExpression(omega=omega, r=r, domain=mesh)

randomFieldProj = fe.project(randomFieldVExpression, V, mesh=mesh)
W = fe.TensorFunctionSpace(mesh, 'P', deg)
def square_boundary(x, on_boundary):
    return on_boundary and (fe.near(x[0], 0) or fe.near(x[0], 0.32) or fe.near(x[1], 0) or fe.near(x[1], 0.32))

bc_jacobian_projection = fe.DirichletBC(W, fe.Constant(((1, 0), (0, 1))), square_boundary) 
jacobianProj = fe.project(fe.grad(randomFieldProj), W, bcs=bc_jacobian_projection)

J_inv_T = J_minus_TExpression(jacobianProj, domain=mesh)
J_helper1 = J_helper1Expression(jacobianProj, domain=mesh)
J_helper2 = J_helper2Expression(jacobianProj, domain=mesh)
det_J = J_determinantExpression(jacobianProj, domain=mesh)
inv_det_J = J_inv_determinantExpression(jacobianProj, domain=mesh)


#! just for now
V_scalar = fe.FunctionSpace(mesh, 'P', deg)
c = fe.plot(fe.project(det_J, V_scalar))
plt.colorbar(c)
plt.show()
#! just for now

left_integrand = left_prefactor * det_J * fe.inner(fe.dot(J_inv_T, fe.grad(u)), fe.dot(J_inv_T, fe.grad(v)))

right_integrand = right_prefactor * inv_det_J * \
                  (fe.dot(J_helper1, fe.grad(u)[:, 0]) + fe.dot(J_helper2, fe.grad(u)[:, 1])) * \
                  (fe.dot(J_helper1, fe.grad(v)[:, 0]) + fe.dot(J_helper2, fe.grad(v)[:, 1]))


a = (left_integrand + right_integrand) * fe.dx
# Right-hand side of weak form
L = fe.inner(b,v) * det_J * fe.dx + det_J * fe.dot(g,v) * ds(3)

# Solve Galerkin system
fe.solve(a==L, u_hat_sol, bc_left)




# Plots
plt.figure(figsize=(16, 8))

# Plot the mesh and boundary points
plt.subplot(2, 3, 1)
fe.plot(mesh, title='Reference Mesh')
plt.scatter(inner_circle_boundary_points[:, 0], inner_circle_boundary_points[:, 1], color='cyan', s=10, label='Circle Boundary Points')
plt.scatter(left_boundary_points[:, 0], left_boundary_points[:, 1], color='blue', s=10, label='Left Boundary Points')
plt.scatter(right_boundary_points[:, 0], right_boundary_points[:, 1], color='green', s=10, label='Right Boundary Points')
plt.title('Mesh with Boundary Points')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper right')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot perturbed mesh
plt.subplot(2, 3, 4)
fe.plot(perturbed_mesh, title='Perturbed Mesh')
plt.scatter(left_boundary_points[:, 0], left_boundary_points[:, 1], color='blue', s=10, label='Left Boundary Points')
plt.scatter(right_boundary_points[:, 0], right_boundary_points[:, 1], color='green', s=10, label='Right Boundary Points')
plt.title('Perturbed Mesh')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper right')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot the solution û
plt.subplot(2, 3, 2)
c = fe.plot(u_hat_sol, title=r'Displacement $\hat{u}(x, \omega)$')
plt.colorbar(c)
plt.title(r'Displacement $\hat{u}(x, \omega)$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot the solution u
V_perturbed = fe.VectorFunctionSpace(perturbed_mesh, "P", deg)
u_sol = fe.Function(V_perturbed)
u_sol.vector()[:] = u_hat_sol.vector()[:] 
plt.subplot(2, 3, 5)
c = fe.plot(u_sol, title=r'Displacement $u(x, \omega)$')
plt.colorbar(c)
plt.title(r'Displacement $u(x, \omega)$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot Sigma(û)
def sigma(u):
    return left_prefactor*fe.grad(u) + right_prefactor*fe.div(u)*fe.Identity(2)
# def sigma(u): # Stress tensor
#     return left_prefactor * fe.dot(J_inv_T, fe.grad(u)) + right_prefactor * inv_det_J * (fe.dot(J_helper1, fe.grad(u)[:, 0]) + fe.dot(J_helper2, fe.grad(u)[:, 1])) * fe.Identity(2)

sigma_proj = fe.project(sigma(u_hat_sol)[:, 0], V)
plt.subplot(2, 3, 3)
c = fe.plot(sigma_proj, title=r'$\sigma(\hat{u}) \cdot e_1$')
plt.colorbar(c)
plt.title(r'$\sigma(\hat{u}) \cdot e_1$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot Sigma(u)
sigma_perturbed_proj = fe.Function(V_perturbed)
sigma_perturbed_proj.vector()[:] = sigma_proj.vector()[:]
plt.subplot(2, 3, 6)
c = fe.plot(sigma_perturbed_proj, title=r'$\sigma(u) \cdot e_1$')
plt.colorbar(c)
plt.title(r'$\sigma(u) \cdot e_1$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

plt.tight_layout()
plt.show()


print(f"sigma_proj(fe.Point(0.32, 0.32)): {sigma_proj(fe.Point(0.32, 0.32))}")
print(f"sigma_proj(fe.Point(0.32, 0.28)): {sigma_proj(fe.Point(0.32, 0.28))}")
print(f"sigma_proj(fe.Point(0.32, 0.24)): {sigma_proj(fe.Point(0.32, 0.24))}")
print(f"sigma_proj(fe.Point(0.32, 0.20)): {sigma_proj(fe.Point(0.32, 0.20))}")
print(f"sigma_proj(fe.Point(0.32, 0.16)): {sigma_proj(fe.Point(0.32, 0.16))}")
print(f"sigma_proj(fe.Point(0.32, 0.12)): {sigma_proj(fe.Point(0.32, 0.12))}")
print(f"sigma_proj(fe.Point(0.32, 0.08)): {sigma_proj(fe.Point(0.32, 0.08))}")
print(f"sigma_proj(fe.Point(0.32, 0.04)): {sigma_proj(fe.Point(0.32, 0.04))}")
print(f"sigma_proj(fe.Point(0.32, 0.0)): {sigma_proj(fe.Point(0.32, 0.0))}")

print(f"jacobianProj(np.array([0.32, 0.32])): {jacobianProj(np.array([0.32, 0.32]))}")
print(f"jacobianProj(np.array([0.32, 0.28])): {jacobianProj(np.array([0.32, 0.28]))}")
print(f"jacobianProj(np.array([0.32, 0.24])): {jacobianProj(np.array([0.32, 0.24]))}")
print(f"jacobianProj(np.array([0.32, 0.20])): {jacobianProj(np.array([0.32, 0.20]))}")
print(f"jacobianProj(np.array([0.32, 0.16])): {jacobianProj(np.array([0.32, 0.16]))}")
print(f"jacobianProj(np.array([0.32, 0.12])): {jacobianProj(np.array([0.32, 0.12]))}")
print(f"jacobianProj(np.array([0.32, 0.08])): {jacobianProj(np.array([0.32, 0.08]))}")
print(f"jacobianProj(np.array([0.32, 0.04])): {jacobianProj(np.array([0.32, 0.04]))}")
print(f"jacobianProj(np.array([0.32, 0.0])): {jacobianProj(np.array([0.32, 0.0]))}")

print(f"det_J(np.array([0.32, 0.32])): {det_J(np.array([0.32, 0.32]))}")
print(f"det_J(np.array([0.32, 0.28])): {det_J(np.array([0.32, 0.28]))}")
print(f"det_J(np.array([0.32, 0.24])): {det_J(np.array([0.32, 0.24]))}")
print(f"det_J(np.array([0.32, 0.20])): {det_J(np.array([0.32, 0.20]))}")
print(f"det_J(np.array([0.32, 0.16])): {det_J(np.array([0.32, 0.16]))}")
print(f"det_J(np.array([0.32, 0.12])): {det_J(np.array([0.32, 0.12]))}")
print(f"det_J(np.array([0.32, 0.08])): {det_J(np.array([0.32, 0.08]))}")
print(f"det_J(np.array([0.32, 0.04])): {det_J(np.array([0.32, 0.04]))}")
print(f"det_J(np.array([0.32, 0.0])): {det_J(np.array([0.32, 0.0]))}")

print(f"perturbation_function(np.array([0.32, 0.32])): {perturbation_function(np.array([0.32, 0.32]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.28])): {perturbation_function(np.array([0.32, 0.28]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.24])): {perturbation_function(np.array([0.32, 0.24]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.20])): {perturbation_function(np.array([0.32, 0.20]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.16])): {perturbation_function(np.array([0.32, 0.16]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.12])): {perturbation_function(np.array([0.32, 0.12]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.08])): {perturbation_function(np.array([0.32, 0.08]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.04])): {perturbation_function(np.array([0.32, 0.04]), omega, r)}")
print(f"perturbation_function(np.array([0.32, 0.0])): {perturbation_function(np.array([0.32, 0.0]), omega, r)}")

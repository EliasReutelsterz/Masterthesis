from __future__ import print_function
import fenics as fe
import mshr
import numpy as np
import matplotlib.pyplot as plt
import os

from model_sp_3_helper import *

current_dir = os.path.dirname(os.path.abspath(__file__))


# Design parameters
mesh_resolution = 12

# Model parameters
a = 0.32
r = 0.02
nu = 0.29
rho = 7850
grav = 9.80665
b = fe.Constant((0, -rho * grav))
q = 60*1E6

mu_E_prime = 26.011
sigma_E_prime = 0.149

# Mesh
bottom_left_corner = fe.Point(0, 0)
top_right_corner = fe.Point(a, a)
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


# Save mesh
mesh_path = os.path.join(current_dir, 'data_output/elasticity_problem_1_mesh.xml')
fe.File(mesh_path) << mesh

deg = 1
V = fe.VectorFunctionSpace(mesh, 'P', deg)
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
u_sol = fe.Function(V)

# Boundary conditions
ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
bc_left = fe.DirichletBC(V, fe.Constant((0,0)), left_boundary_function)

g = fe.Constant((q,0))

# E random field
randomFieldE = calculate_randomFieldE(mesh=mesh, mu=mu_E_prime, sigma=sigma_E_prime)
xi = np.random.normal(0, 1, size=randomFieldE.J)
randomFieldEExpression = RandomFieldEExpression(randomFieldE=randomFieldE, xi=xi)

mu = fe.Expression('E / (2 * (1 + nu))', degree=1, E=randomFieldEExpression, nu=nu, domain=mesh)
right_prefactor = fe.Expression('E / (2 * (1 - nu))', degree=1, E=randomFieldEExpression, nu=nu, domain=mesh)

a = fe.inner(fe.grad(u), fe.grad(mu * v))*fe.dx + fe.div(u)*fe.div(right_prefactor * v)*fe.dx
# Right-hand side of weak form
L = fe.inner(b,v)*fe.dx + fe.dot(g,v)*ds(3)

# Solve Galerkin system
fe.solve(a==L, u_sol, bc_left)




# Plots
plt.figure(figsize=(12, 8))

# Plot the mesh and boundary points
plt.subplot(2, 2, 1)
fe.plot(mesh, title='Mesh')
plt.scatter(inner_circle_boundary_points[:, 0], inner_circle_boundary_points[:, 1], color='cyan', s=10, label='Circle Boundary Points')
plt.scatter(left_boundary_points[:, 0], left_boundary_points[:, 1], color='blue', s=10, label='Left Boundary Points')
plt.scatter(right_boundary_points[:, 0], right_boundary_points[:, 1], color='green', s=10, label='Right Boundary Points')
plt.title('Mesh with Boundary Points')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper right')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)
    

# Plot random field E
W = fe.FunctionSpace(mesh, 'P', deg)
E_proj = fe.project(randomFieldEExpression, W)
plt.subplot(2, 2, 2)
c = fe.plot(E_proj, title="Random Field E")
plt.colorbar(c)
plt.title('Random Field E')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# Plot the solution u
plt.subplot(2, 2, 3)
c = fe.plot(u_sol, title='Displacement')
plt.colorbar(c)
plt.title('Displacement')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)


# Plot sigma * n (sigma differs to sp_1)
def sigma(u): # Stress tensor
    return mu*fe.grad(u) + right_prefactor*fe.div(u)*fe.Identity(2)

sigma_proj = fe.project(sigma(u_sol)[:, 0], V)
plt.subplot(2, 2, 4)
c = fe.plot(sigma_proj, title='Sigma * e_1')
plt.colorbar(c)
plt.title('Sigma * e_1')
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
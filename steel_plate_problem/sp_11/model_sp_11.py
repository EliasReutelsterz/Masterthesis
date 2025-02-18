from __future__ import print_function
# from dolfin import *
from fenics import *
from mshr import *
import numpy as np
'''
Solver for the stationary Navier-Lame equations (DISPLACEMENT formulation) 
 
Domain: 		Lego beam
Boundary condition: 	clamped at left end, traction condition at right end
Body force:             Deformation of beam under its own weight
Material:               PVC plastic

u: 		Displacement vector (3 x 1)
sigma: 	        Stress tensor (3 x 3)
epsilon: 	Strain tensor (3 x 3)

E:		Young's modulus (1 x 1)
nu:        	Poisson's ratio (1 x 1)
rho:		Density (1 x 1 )

Finite elements for u: vector-valued, linear Lagrange in each component

Acknowledgement: This file is based on a python script provided by Markus Muhr (M2, TUM)
'''
# (1) Define problem parameters
    # Material parameters
rho = 1.45*1E3
E = 0.0065*1E9
nu = 0.41
#nu = 0.4999999  # Locking

    # Calculate the Lame parameters
mu = E/(2*(1+nu))
lambda_ = nu*E/((1+nu)*(1-2*nu))
    # Body force
F = 9.81  # gravitational acceleration 
f = Constant((0,0,-rho*F))

# (2) Define geometry and mesh
    # Create mesh
mesh = Mesh('lego_beam.xml')
    # Store mesh
vtkfile = File('lego/Mesh.pvd')
vtkfile << mesh

    # Specify clamped and traction boundary
xmin = mesh.coordinates()[:, 0].min()
xmax = mesh.coordinates()[:, 0].max()

print(np.min(xmin), np.max(xmax))

def clamped_bndr(x, on_boundary):
    return on_boundary and near(x[0],xmin)
def traction_bndr(x, on_boundary):
    return on_boundary and near(x[0],xmax)

    # Mark boundary parts
boundary_markers = MeshFunction('size_t', mesh, 1)
boundary_markers.set_all(9999)

class BoundaryXmin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], xmin)

class BoundaryXmax(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], xmax)

bxmin = BoundaryXmin()
bxmax = BoundaryXmax()
bxmin.mark(boundary_markers, 0)
bxmax.mark(boundary_markers, 1)

    # Redefine boundary integration measure
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# (3) Define finite element space for displacement
    # Finite Element degree
deg = 1
      # Define function space
V = VectorFunctionSpace(mesh, 'P', deg)

    # Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
    # Define solution
u_sol = Function(V)

# (4) Set up boundary data
    # Define clamped boundary value (u = 0-vector)
bv = Constant((0,0,0))
    # Define boundary condition
bc = DirichletBC(V, bv, clamped_bndr)
    # Define traction vector
g = Constant((0,0,-5*1E3))

# (5) Define variational problem
    # Strain tensor
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)
    # Stress tensor
def sigma(u):
    return 2*mu*epsilon(u) + lambda_*div(u)*Identity(3)

    # Left-hand side of weak form
a = 2*mu*inner(epsilon(u), epsilon(v))*dx + lambda_*div(u)*div(v)*dx
    # Right-hand side of weak form
L = inner(f,v)*dx + dot(g,v)*ds(1)

# (6) Solve Galerkin system
solve(a==L, u_sol, bc)

# (7) Store results
    # Displacement
u_sol.rename('Displacement', 'u')
    # Store in .pvd file for use with Paraview
vtkfile = File('lego/Beam_deformation.pvd')
vtkfile << u_sol

#! check boundary condition for right boundary
right_boundary_points = np.array([p for p in mesh.coordinates() if near(p[0], xmax)])
sigma_proj = project(sigma(u_sol), TensorFunctionSpace(mesh, 'DG', 0))
for point in right_boundary_points:
    print(f"point: {point}")
    print(f"sigma_proj(point)(3, 3)[:, 0]: {sigma_proj(point).reshape(3, 3)[:, 0]}")
print(f"g: {[0,0,-5*1E3]}")

    # Stress 
s = sigma(u_sol) - (1./3)*tr(sigma(u_sol))*Identity(3)  # deviatoric stress
    # Evaluate Von-Mises stress
von_Mises = sqrt(3./2*inner(s, s))
V_scalar = FunctionSpace(mesh, 'P', deg+1)
    # Project Von-Mises stress into (scalar-valued) function space - L2 projection
von_Mises = project(von_Mises, V_scalar)
    
    # Store in .pvd file for use with Paraview
vtkfile = File('lego/VonMisesStress.pvd')
von_Mises.rename('von_Mises stress', 'sigma')
vtkfile << von_Mises



#! test:

from scipy.interpolate import griddata


# Project stress (as you did before)
W_tensor = TensorFunctionSpace(mesh, "P", 1)
sigma_proj = project(sigma(u_sol), W_tensor)


right_boundary_points = np.array([p for p in mesh.coordinates() if near(p[0], xmax)])
first_components = []
second_components = []
third_components = []
for point in right_boundary_points:
    sigma_val = sigma_proj(point).reshape(3, 3)[:, 0]
    first_components.append(sigma_val[0])
    second_components.append(sigma_val[1])
    third_components.append(sigma_val[2])


# Reshape the components into 2D arrays
right_boundary_y = right_boundary_points[:, 1]
right_boundary_z = right_boundary_points[:, 2]

# Create a grid for the contour plot
y_unique = np.unique(right_boundary_y)
z_unique = np.unique(right_boundary_z)
Y, Z = np.meshgrid(y_unique, z_unique)

# Interpolate the components onto the grid
first_components_grid = griddata((right_boundary_y, right_boundary_z), first_components, (Y, Z), method='linear')
second_components_grid = griddata((right_boundary_y, right_boundary_z), second_components, (Y, Z), method='linear')
third_components_grid = griddata((right_boundary_y, right_boundary_z), third_components, (Y, Z), method='linear')


# plot contourf of (y,z) and components of sigma
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.contourf(Y, Z, first_components_grid, 100, cmap='viridis')
plt.colorbar()
plt.title(r"First component $\sigma \cdot n(x)$, $g[0] = 0$")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 2)
plt.contourf(Y, Z, second_components_grid, 100, cmap='viridis')
plt.colorbar()
plt.title(r"Second component $\sigma \cdot n(x)$, $g[1] = 0$")
plt.xlabel("y")
plt.ylabel("z")

plt.subplot(1, 3, 3)
plt.contourf(Y, Z, third_components_grid, 100, cmap='viridis')
plt.colorbar()
plt.title(r"Third component $\sigma \cdot n(x)$, $g[2] = -5*1E3$")
plt.xlabel("y")
plt.ylabel("z")

plt.tight_layout()
plt.show()


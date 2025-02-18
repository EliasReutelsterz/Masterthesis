from dolfin import *
from mshr import *

L, R = 1., 0.1
N = 200 # mesh density

domain = Rectangle(Point(0.,0.), Point(L, L)) - Circle(Point(0., 0.), R)
mesh = generate_mesh(domain, N)

Ex, Ey, nuxy, Gxy = 100., 10., 0.3, 5.
S = as_matrix([[1./Ex,-nuxy/Ex,0.],[-nuxy/Ex,1./Ey,0.],[0.,0.,1./Gxy]])
C = inv(S)

def eps(v):
    return sym(grad(v))
def strain2voigt(e):
    """e is a 2nd-order tensor, returns its Voigt vectorial representation"""
    return as_vector([e[0,0],e[1,1],2*e[0,1]])
def voigt2stress(s):
    """
    s is a stress-like vector (no 2 factor on last component)
    returns its tensorial representation
    """
    return as_tensor([[s[0], s[2]],
                     [s[2], s[1]]])
def sigma(v):
    return voigt2stress(dot(C, strain2voigt(eps(v))))

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], L) and on_boundary
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary

# exterior facets MeshFunction
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
Top().mark(facets, 1)
Left().mark(facets, 2)
Bottom().mark(facets, 3)
ds = Measure('ds', subdomain_data=facets)

# Define function space
V = VectorFunctionSpace(mesh, 'Lagrange', 2)

# Define variational problem
du = TrialFunction(V)
u_ = TestFunction(V)
u = Function(V, name='Displacement')
a = inner(sigma(du), eps(u_))*dx

# uniform traction on top boundary
T0 = 0
T1 = 1e-3
T = Constant((T0, T1))
l = dot(T, u_)*ds(1)

# symmetry boundary conditions
bc = [DirichletBC(V.sub(0), Constant(0.), facets, 2),
      DirichletBC(V.sub(1), Constant(0.), facets, 3)]

solve(a == l, u, bc)

import matplotlib.pyplot as plt
p = plot(sigma(u)[1, 1]/T[1], mode='color')
plt.colorbar(p)
plt.title(r"$\sigma_{yy}$",fontsize=26)
plt.show()



#! OWN CODE
Vsig = TensorFunctionSpace(mesh, "P", degree=1)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u), Vsig))
import numpy as np
for top_boundary_point in [Point(x[0], x[1]) for x in mesh.coordinates() if np.isclose(x[1], L)]:
    sigma_tensor = np.array(sig(top_boundary_point)).reshape(2, 2)
    print(f"sigma_tensor[:, 1]: {sigma_tensor[:, 1]}")
    print(f"T: [{T0}, {T1}]")

top_boundary_points = np.linspace(start=0, stop=L, num=100)
sigma_u_right_boundary = np.array([np.array(sig(np.array([point, L]))).reshape(2, 2)[:, 1] for point in top_boundary_points])


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(top_boundary_points, sigma_u_right_boundary[:, 0], label=r'First component $\sigma(u) \cdot e_2$')
plt.axhline(T0, label='Reference for T[0]={T0}', color="red")
plt.legend()
plt.xlabel(r'$x_1$')

plt.subplot(1, 2, 2)
plt.plot(top_boundary_points, sigma_u_right_boundary[:, 1], label=r'Second component $\sigma(u) \cdot e_2$')
plt.axhline(T1, label=f'Reference for T[1]={T1}', color="red")
plt.xlabel(r'$x_1$')
plt.legend()
plt.tight_layout()
plt.show()

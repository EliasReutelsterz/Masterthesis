#
# ..    # gedit: set fileencoding=utf8 :
# .. raw:: html
#
#  <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><p align="center"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png"/></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a></p>
#
#
# .. _LinearElasticity2D:
#
# =========================
#  2D linear elasticity
# =========================
#
#
# Introduction
# ------------
#
# In this first numerical tour, we will show how to compute a small strain solution for
# a 2D isotropic linear elastic medium, either in plane stress or in plane strain,
# in a tradtional displacement-based finite element formulation. The corresponding
# file can be obtained from :download:`2D_elasticity.py`.
#
# .. seealso::
#
#  Extension to 3D is straightforward and an example can be found in the :ref:`ModalAnalysis` example.
#
# We consider here the case of a cantilever beam modeled as a 2D medium of dimensions
# :math:`L\times  H`. Geometrical parameters and mesh density are first defined
# and the rectangular domain is  generated using the ``RectangleMesh`` function.
# We also choose a criss-crossed structured mesh::

from dolfin import *
import mshr

L = 25.
H = 25.
Nx = 32
Ny = 10
domain = mshr.Rectangle(Point(0., 0.), Point(L, H)) - mshr.Circle(Point(L/2, H/2), 0.2)
mesh = mshr.generate_mesh(domain, Nx)
#! mesh = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny, "crossed") - mshr.Circle(Point(0.5, 0.5), 0.1)



# Constitutive relation
# ---------------------
#
# We now define the material parameters which are here given in terms of a Young's
# modulus :math:`E` and a Poisson coefficient :math:`\nu`. In the following, we will
# need to define the constitutive relation between the stress tensor :math:`\boldsymbol{\sigma}`
# and the strain tensor :math:`\boldsymbol{\varepsilon}`. Let us recall
# that the general expression of the linear elastic isotropic constitutive relation
# for a 3D medium is given by:
#
# .. math::
#  \boldsymbol{\sigma} = \lambda \text{tr}(\boldsymbol{\varepsilon})\mathbf{1} + 2\mu\boldsymbol{\varepsilon}
#  :label: constitutive_3D
#
# for a natural (no prestress) initial state where the LamÃ© coefficients are given by:
#
# .. math::
#  \lambda = \dfrac{E\nu}{(1+\nu)(1-2\nu)}, \quad \mu = \dfrac{E}{2(1+\nu)}
#  :label: Lame_coeff
#
# In this demo, we consider a 2D model either in plane strain or in plane stress conditions.
# Irrespective of this choice, we will work only with a 2D displacement vector :math:`\boldsymbol{u}=(u_x,u_y)`
# and will subsequently define the strain operator ``eps`` as follows::

def eps(v):
    return sym(grad(v))

# which computes the 2x2 plane components of the symmetrized gradient tensor of
# any 2D vectorial field. In the plane strain case, the full 3D strain tensor is defined as follows:
#
# .. math::
#  \boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_{xx} & \varepsilon_{xy} & 0\\
#  \varepsilon_{xy} & \varepsilon_{yy} & 0 \\ 0 & 0 & 0\end{bmatrix}
#
# so that the 2x2 plane part of the stress tensor is defined in the same way as for the 3D case
# (the out-of-plane stress component being given by :math:`\sigma_{zz}=\lambda(\varepsilon_{xx}+\varepsilon_{yy})`.
#
# In the plane stress case, an out-of-plane strain component :math:`\varepsilon_{zz}`
# must be considered so that :math:`\sigma_{zz}=0`. Using this condition in the
# 3D constitutive relation, one has :math:`\varepsilon_{zz}=-\dfrac{\lambda}{\lambda+2\mu}(\varepsilon_{xx}+\varepsilon_{yy})`.
# Injecting into :eq:`constitutive_3D`, we have for the 2D plane stress relation:
#
# .. math::
#  \boldsymbol{\sigma} = \lambda^* \text{tr}(\boldsymbol{\varepsilon})\mathbf{1} + 2\mu\boldsymbol{\varepsilon}
#
# where :math:`\boldsymbol{\sigma}, \boldsymbol{\varepsilon}, \mathbf{1}` are 2D tensors and with
# :math:`\lambda^* = \dfrac{2\lambda\mu}{\lambda+2\mu}`. Hence, the 2D constitutive relation
# is identical to the plane strain case by changing only the value of the LamÃ© coefficient :math:`\lambda`.
# We can then have::

E = Constant(1e5)
nu = Constant(0.3)
model = "plane_stress"

mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
if model == "plane_stress":
    lmbda = 2*mu*lmbda/(lmbda+2*mu)

def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

# Right boundary
import numpy as np
class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and np.isclose(x[0], L)         
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(9999)
right_boundary = RightBoundary()
right_boundary.mark(boundary_markers, 2)
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

rho_g = 1e-3
f = Constant((0, -rho_g))
q = 60 * 1e6
g = Constant((q, 0))

V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
du = TrialFunction(V)
u_ = TestFunction(V)
a = inner(sigma(du), eps(u_))*dx
l = inner(f, u_)*dx + dot(g, u_) * ds(2)


def left(x, on_boundary):
    return near(x[0], 0.)

bc = DirichletBC(V, Constant((0.,0.)), left)

u = Function(V, name="Displacement")
solve(a == l, u, bc)

plot(u)
import matplotlib.pyplot as plt
plt.show()


print("Maximal deflection:", -u(L,H/2.)[1])
print("Beam theory deflection:", float(3*rho_g*L**4/2/E/H**3))


Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
sig = Function(Vsig, name="Stress")
sig.assign(project(sigma(u), Vsig))
eps_func = Function(Vsig, name="Stress")
eps_func.assign(project(eps(u), Vsig))
points_right_boundary = [Point(x[0], x[1]) for x in mesh.coordinates() if np.isclose(x[0], L)]
for point in points_right_boundary:
    print(f"Stress at ({point[0]}, {point[1]}): {np.array(sig(point)).reshape(2, 2)[:, 0]}")
    print(f"Strain at ({point[0]}, {point[1]}): {np.array(eps_func(point)).reshape(2, 2)[:, 0]}")
    print(f"Stress times u: {np.dot(np.array(sig(point)).reshape(2, 2), np.array(u(point)))}")
    print(f"Stress times grad u: {np.dot(np.array(sig(point)).reshape(2, 2), np.array(eps_func(point)).reshape(2, 2))}")

#! facetnormals

n = FacetNormal(mesh)
V = VectorFunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(u,v)*ds
l = inner(n, v)*ds
A = assemble(a, keep_diagonal=True)
L = assemble(l)

A.ident_zeros()
nh = Function(V)

solve(A, nh.vector(), L)

for point in points_right_boundary:
    print(f"Normal vector at [{point[0]}, {point[1]}]: {nh(point)}")
    print(f"Stress * normal: {np.dot(np.array(sig(point)).reshape(2, 2), np.array(nh(point)))}")
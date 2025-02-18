import fenics as fe
import numpy as np
import mshr
import matplotlib.pyplot as plt


def solve_model(mesh_resolution: int = 32):
    # Material parameters
    E = np.exp(26.011)
    q = 60 * 1e6
    g = fe.Constant((q, 0))
    rho = 7850
    g_gravity = 9.80665
    b = fe.Constant((0, - rho * g_gravity))
    nu = 0.29

    # Mesh
    domain_ref = mshr.Rectangle(fe.Point(0, 0), fe.Point(0.32, 0.32)) - mshr.Circle(fe.Point(0.16, 0.16), 0.02)
    mesh = mshr.generate_mesh(domain_ref, mesh_resolution)
    mesh = fe.Mesh(mesh)
    
    # Mark boundaries
    class RightBoundary(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and np.isclose(x[0], 0.32)
            
    class LeftBoundary(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and np.isclose(x[0], 0)

    boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    boundary_markers.set_all(9999)
    right_boundary = RightBoundary()
    right_boundary.mark(boundary_markers, 2)
    left_boundary = LeftBoundary()
    left_boundary.mark(boundary_markers, 1)
    ds = fe.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    
    # Left boundary
    V = fe.VectorFunctionSpace(mesh, "P", 1)
    bc_gamma1 = fe.DirichletBC(V, fe.Constant((0, 0)), boundary_markers, 1)
    
    def sigma(u):
        eps = fe.sym(fe.grad(u))
        mu = E/2/(1+nu)
        lmbda = E*nu/(1+nu)/(1-2*nu)
        lmbda = 2*mu*lmbda/(lmbda+2*mu)
        return lmbda*fe.tr(eps)*fe.Identity(2) + 2.0*mu*eps
        #! currently not used
        u_grad = fe.grad(u)
        return fe.as_matrix([[E/(1-nu**2) * (u_grad[0, 0] + nu * u_grad[1, 1]),
                              E/(2*(1+nu)) * (u_grad[0, 1] + u_grad[1, 0])],
                             [E/(2*(1+nu)) * (u_grad[0, 1] + u_grad[1, 0]),
                              E/(1-nu**2) * (u_grad[1, 1] + nu * u_grad[0, 0])]])

    # Varational formulation
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    a = fe.inner(sigma(u), fe.grad(v)) * fe.dx
    l = fe.dot(b, v) * fe.dx + fe.dot(g, v) * ds(2)

    # Solve
    u_sol = fe.Function(V)
    fe.solve(a == l, u_sol, bc_gamma1)

    # Plot solution
    fe.plot(u_sol)
    plt.show()

    W = fe.TensorFunctionSpace(mesh, "P", 1)

    sigma_proj = fe.project(sigma(u_sol), W)

    right_boundary_test_points = [fe.Point(x[0], x[1]) for x in mesh.coordinates() if np.isclose(x[0], 0.32)]
    n = np.array([1, 0])  # Outward normal vector on the right boundary

    for point in right_boundary_test_points:
        sigma_tensor = sigma_proj(point)
        sigma_evaluated = np.array(sigma_tensor).reshape(2, 2)
        traction_calculated = np.dot(sigma_evaluated, n)
        print(f"Point: {point[0]}, {point[1]}")
        print(f"Calculated Traction: {traction_calculated}")
        print(f"Applied Traction: [{q}, {0}]")
        print(f"Absolute Difference: {np.abs(traction_calculated - np.array([q, 0]))}")


    fe.plot(sigma_proj)
    plt.show()


    # Plot sigma(u) and sigma_hat(u_hat) along right boundary
    right_boundary_points = np.linspace(start=0, stop=0.32, num=100)
    sigma_u_right_boundary = np.array([np.array(sigma_proj(np.array([0.32, point]))).reshape(2, 2)[:, 0] for point in right_boundary_points])
    

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


if __name__ == "__main__":
    solve_model()
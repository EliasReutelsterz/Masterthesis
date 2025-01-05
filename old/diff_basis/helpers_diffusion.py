from helpers import *

class RandomFieldZ():
    def __init__(self, eigenvalues, eigenvectors, basis_functions, N, J):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J

    def __call__(self, x, xi):
        return sum([np.sqrt(self.eigenvalues[j]) * sum([self.eigenvectors[k, j] * self.basis_functions[k].function(x) for k in range(self.N)]) * xi[j] for j in range(len(xi))])

def z_calculate_random_field_eigenpairs(mesh_resolution, z_cov):  
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "P", 3)
    V_Vector = fe.VectorFunctionSpace(mesh, "P", 3)

    basis_functions = []
    basis_functions_grads = []

    for i in range(V.dim()):
        basis_function = fe.Function(V)
        basis_function.vector()[i] = 1
        for vertex in fe.vertices(mesh):
            if abs(basis_function(vertex.point().array()[:2]) - 1) < 1e-6:
                basis_function.set_allow_extrapolation(True)
                grad = fe.project(fe.grad(basis_function), V_Vector)
                grad.set_allow_extrapolation(True)
                supp_cells = [cell for cell in fe.cells(mesh) if basis_function(cell.midpoint().array()[:2]) != 0]
                basis_functions.append(BasisFunction(basis_function, supp_cells))
                basis_functions_grads.append(BasisFunction(grad, supp_cells))
    
    N = len(basis_functions)

    C = np.zeros((N, N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if j <= i:
                # Here we use that each block is symmetric because of the symmetry of the covariance functions
                C[i, j] = C[j, i] = get_C_entry(z_cov, basis_function_i, basis_function_j)
    
    M = np.zeros((N, N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if i >= j:
                integrand = basis_function_j.function * basis_function_i.function * fe.dx
                M[i, j] = M[j, i] = fe.assemble(integrand)
        
    J = N # Number of eigenvectors -> J = N is maximum
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

    return RandomFieldZ(sorted_eigenvalues, sorted_eigenvectors, basis_functions, N, J)


def a_hat_random_field(x, randomFieldV: RandomFieldV, randomFieldZ: RandomFieldZ, xi_v, xi_z):
    # log-normal random field with a_hat = exp(z(V(x)))
    # mean of z = 0
    x_hat = randomFieldV(x, xi_v)
    z = randomFieldZ(x_hat, xi_z)
    a = np.exp(z)
    # print(f"x: {x}, x_hat: {x_hat}, z: {z}, a: {a}")
    return a

class BExpression(fe.UserExpression):
    def __init__(self, randomFieldV, jacobianV, randomFieldZ, xi_v, xi_z, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldV = randomFieldV
        self.jacobianV = jacobianV
        self.randomFieldZ = randomFieldZ
        self.xi_v = xi_v
        self.xi_z = xi_z

    def eval(self, values, x):
        J_x = self.jacobianV(x, self.xi_v)
        inv_JTJ = np.linalg.inv(J_x.T @ J_x)
        det_J = np.linalg.det(J_x)
        a_hat = a_hat_random_field(x, self.randomFieldV, self.randomFieldZ, self.xi_v, self.xi_z)
        B_x = a_hat * inv_JTJ * det_J
        values[0] = B_x[0, 0]
        values[1] = B_x[0, 1]
        values[2] = B_x[1, 0]
        values[3] = B_x[1, 1]

    def value_shape(self):
        return (2, 2)
    
def solve_diffusion_poisson_for_given_sample(mesh_resolution, f, randomFieldV, jacobianV, randomFieldZ, xi_v, xi_z):
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 3)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    B_expr = BExpression(randomFieldV, jacobianV, randomFieldZ, xi_v, xi_z, degree=2)
    a = fe.inner(fe.dot(B_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV, xi_v, degree=2)
    L = f * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    return u_sol
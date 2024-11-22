from helpers import *

class RandomFieldZ():
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array, basis_functions: list[ConstBasisFunction], N: int, J: int):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis_functions = basis_functions
        self.N = N
        self.J = J

    def __call__(self, x, xi):
        index_supported_basis_function = self.find_supported_basis_function(x)
        if index_supported_basis_function is None:
            return 0
        return sum([np.sqrt(self.eigenvalues[j]) * self.eigenvectors[index_supported_basis_function, j] * xi[j] for j in range(len(xi))])
    
    def find_supported_basis_function(self, x):
        for i, basis_function in enumerate(self.basis_functions):
            if basis_function.function(x) == 1:
                return i
        return None


def z_calculate_random_field_eigenpairs(mesh_resolution, z_cov):  
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "DG", 0)
    N = V.dim()

    basis_functions = []
    for i in range(N):
        basis_function = fe.Function(V)
        basis_function.vector()[i] = 1.0
        basis_function.set_allow_extrapolation(True)

        cell_index = V.dofmap().cell_dofs(i)[0]
        cell = fe.Cell(mesh, cell_index)
        vertex_coords = np.array([np.array(mesh.coordinates()[vertex]) for vertex in cell.entities(0)])
        basis_functions.append(ConstBasisFunction(basis_function, vertex_coords))
    
    C = np.zeros((N, N))
    for i, basis_function_i in enumerate(basis_functions):
        for j, basis_function_j in enumerate(basis_functions):
            if j <= i:
                # Here we use that each block is symmetric because of the symmetry of the covariance functions
                C[i, j] = C[j, i] = get_C_entry(z_cov, basis_function_i, basis_function_j)
    
    M = np.zeros((N, N))
    for i, basis_function in enumerate(basis_functions):
        integrand = basis_function_i.function * basis_function_i.function * fe.dx
        M[i, i] = fe.assemble(integrand)
        
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
    return a

class BExpression(fe.UserExpression):
    def __init__(self, randomFieldV, jacobianV_fixed_xi, randomFieldZ, xi_v, xi_z, **kwargs):
        super().__init__(**kwargs)
        self.randomFieldV = randomFieldV
        self.jacobianV_fixed_xi = jacobianV_fixed_xi
        self.randomFieldZ = randomFieldZ
        self.xi_v = xi_v
        self.xi_z = xi_z

    def eval(self, values, x):
        J_x = self.jacobianV_fixed_xi(x)
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
    jacobianV_fixed_xi = JacobianVFixedXi(jacobianV, xi_v)
    B_expr = BExpression(randomFieldV, jacobianV_fixed_xi, randomFieldZ, xi_v, xi_z, degree=2)
    a = fe.inner(fe.dot(B_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV_fixed_xi, degree=2)
    L = f * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    return u_sol
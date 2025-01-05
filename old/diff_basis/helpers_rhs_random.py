import numpy as np

from helpers import *


# Helpers Sensitivity Analysis

class RandomRHS(fe.UserExpression):
    # of the form F_1 \cdot \sin(\lVert x \rVert_2) + F_2 \cdot \cos(\lVert x \rVert_2)
    def __init__(self, F1, F2, **kwargs):
        super().__init__(**kwargs)
        self.F1 = F1
        self.F2 = F2
    
    def eval(self, values, x):
        values[0] = self.F1 * np.sin(np.sqrt(x[0]**2 + x[1]**2)) + self.F2 * np.cos(np.sqrt(x[0]**2 + x[1]**2))
    
    def value_shape(self):
        return ()

def solve_poisson_for_given_sample_rhs_random(mesh_resolution, jacobianV, xi, F_samples):
    f_expr = RandomRHS(F_samples[0], F_samples[1], degree=2)
    mesh = mshr.generate_mesh(DOMAIN, mesh_resolution)
    V = fe.FunctionSpace(mesh, "CG", 3)
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    A_expr = AExpression(jacobianV, xi, degree=2)
    a = fe.inner(fe.dot(A_expr, fe.grad(u)), fe.grad(v)) * fe.dx
    det_J_expr = detJExpression(jacobianV, xi, degree=2)
    L = f_expr * det_J_expr * v * fe.dx
    bc = fe.DirichletBC(V, DIRICHLET_BC, 'on_boundary')
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bc)
    return u_sol


def double_loop_mc(mc_sample_size, mesh_resolution_fem, P, indices, randomFieldV, jacobianV, size_of_xi):
    # Translate math indices to CS indices
    indices = [index - 1 for index in indices]

    # Implementation of the double loop MC estimation of specific Sobol Index
    mean_sols_point_P = np.zeros(mc_sample_size)
    for i in range(mc_sample_size):
        print(f"First loop: Outer MC loop iteration {i+1} / {mc_sample_size}")
        sols_point_P = []
        xi = np.zeros(size_of_xi)
        F_samples = np.zeros(2)
        for index in indices:
            if index < size_of_xi:
                xi[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
            else:
                F_samples[index - size_of_xi] = np.random.normal(0, 1)
        missing_indices = [i for i in range(size_of_xi + 1) if i not in indices]
        for k in range(mc_sample_size):
            for index in missing_indices:
                if index < size_of_xi:
                    xi[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                else:
                    F_samples[index - size_of_xi] = np.random.normal(0, 1)
            P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
            sols_point_P.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
        mean_sols_point_P[i] = np.mean(sols_point_P)
    var_A = np.var(mean_sols_point_P, ddof=1)
    sols_point_P = []
    for i in range(mc_sample_size):
        print(f"Second loop: {i+1} / {mc_sample_size}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        F_samples = np.random.normal(0, 1, 2)
        P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
        sols_point_P.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
    var = np.var(sols_point_P, ddof=1)
    return var_A / var


def pick_freeze(mc_sample_size, mesh_resolution_fem, P, indices, randomFieldV, jacobianV, size_of_xi):
    # Translate math indices to CS indices
    indices = [index - 1 for index in indices]

    y = []
    y_tilde = []
    for i in range(mc_sample_size):
        print(f"First loop: {i+1} / {mc_sample_size}")
        xi = np.zeros(size_of_xi)
        xi_prime = np.zeros(size_of_xi)
        F_samples = np.zeros(2)
        F_samples_prime = np.zeros(2)
        for index in indices:
            if index < size_of_xi:
                sample = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                xi[index] = sample
                xi_prime[index] = sample
            else:
                sample = np.random.normal(0, 1)
                F_samples[index - size_of_xi] = sample
                F_samples_prime[index - size_of_xi] = sample
        for index in [i for i in range(size_of_xi + 1) if i not in indices]:
            if index < size_of_xi:
                xi[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
                xi_prime[index] = np.random.uniform(-np.sqrt(3), np.sqrt(3))
            else:
                F_samples[index - size_of_xi] = np.random.normal(0, 1)
                F_samples_prime[index - size_of_xi] = np.random.normal(0, 1)

        print(f"indices: {indices}")
        print(f"xi: {xi}")
        print(f"xi_prime: {xi_prime}")
        print(f"F_samples: {F_samples}")
        print(f"F_samples_prime: {F_samples_prime}")

        P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
        y.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
        P_prime_hat = inverse_mapping(P, randomFieldV, xi_prime, mesh_resolution_fem)
        y_tilde.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi_prime, F_samples_prime)(P_prime_hat))
    y_bar = np.mean(y)
    y_tilde_bar = np.mean(y_tilde)
    if mc_sample_size > 1:
        var_A = 1/(mc_sample_size - 1) * sum([(y[i] - y_bar) * (y_tilde[i] - y_tilde_bar) for i in range(mc_sample_size)])
    else:
        var_A = 0

    sols_point_P = []
    for i in range(mc_sample_size):
        print(f"Second loop: {i+1} / {mc_sample_size}")
        xi = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        F_samples = np.random.normal(0, 1, 2)
        P_hat = inverse_mapping(P, randomFieldV, xi, mesh_resolution_fem)
        sols_point_P.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xi, F_samples)(P_hat))
    var = np.var(sols_point_P, ddof=1)
    return var_A / var

def rank(i, sorted_samples, samples):
    return sorted_samples.index(samples[i])
def rank_inv(i, sorted_samples, samples):
    return samples.index(sorted_samples[i])
def P_j(i, sorted_samples, samples):
    if rank(i, sorted_samples, samples) + 2 <= len(samples):
        return rank_inv(rank(i, sorted_samples, samples) + 1, sorted_samples, samples)
    else:
        return rank_inv(0, sorted_samples, samples)

def rank_statistics_permute(samples, y):
    sorted_samples = sorted(samples)
    helper = np.zeros((len(y), 3))
    for i in range(len(y)):
        helper[i, 0] = y[i]
        helper[i, 1] = P_j(i, sorted_samples, samples)
        helper[i, 2] = samples[P_j(i, sorted_samples, samples)]
    sorted_helper = helper[helper[:, 2].argsort()]
    return sorted_helper[:, 0]
    
def rank_statistics(mc_sample_size, mesh_resolution_fem, P, index, randomFieldV, jacobianV, size_of_xi):

    # Translate math index to CS index
    index -= 1

    y = []
    xis = np.zeros((mc_sample_size, size_of_xi))
    F_samples = np.zeros((mc_sample_size, 2))
    for i in range(mc_sample_size):
        print(f"Loop: {i+1} / {mc_sample_size}")
        xis[i, :] = np.random.uniform(-np.sqrt(3), np.sqrt(3), size_of_xi)
        F_samples[i, :] = np.random.normal(0, 1, 2)
        P_hat = inverse_mapping(P, randomFieldV, xis[i, :], mesh_resolution_fem)
        y.append(solve_poisson_for_given_sample_rhs_random(mesh_resolution_fem, jacobianV, xis[i, :], F_samples[i, :])(P_hat))
    if index < size_of_xi:
        y = rank_statistics_permute(xis[:, index].tolist(), y)
        y_bar = np.mean(y)
        V_hat_j = 1/mc_sample_size * sum([y[i] * y[P_j(i, sorted(xis[:, index].tolist()), xis[:, index].tolist())] for i in range(mc_sample_size)]) - y_bar**2
    else:
        y = rank_statistics_permute(F_samples[:, index - size_of_xi].tolist(), y)
        y_bar = np.mean(y)
        V_hat_j = 1/mc_sample_size * sum([y[i] * y[P_j(i, sorted(F_samples[:, index - size_of_xi].tolist()), F_samples[:, index - size_of_xi].tolist())] for i in range(mc_sample_size)]) - y_bar**2
    V_hat = 1/mc_sample_size * sum([y[i]**2 for i in range(mc_sample_size)]) - y_bar**2
    return V_hat_j / V_hat
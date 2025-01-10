import fenics as fe
import numpy as np


def perturbation_function(x: np.array, omega: np.array, r: float) -> np.array:
    x = x - np.array([0.16, 0.16])
    c = np.sqrt(x[0]**2 + x[1]**2)
    x_circ_proj = r/c * x

    theta = np.arctan2(x[1], x[0]) # order has to be y, x as tan(y/x)=theta

    right_left = False
    if -np.pi/4 <= theta <= np.pi/4:
        x_bound_proj = np.array([0.16, 0.16*np.tan(theta)])
        right_left = True
    elif np.pi/4 <= theta <= 3*np.pi/4:
        x_bound_proj = np.array([0.16 / np.tan(theta), 0.16])
    elif theta <= -3*np.pi/4 or theta >= 3*np.pi/4:
        x_bound_proj = np.array([-0.16, -0.16*np.tan(theta)])
        right_left = True
    else:
        x_bound_proj = np.array([-0.16 / np.tan(theta), -0.16])

    h_max = np.sqrt((x_bound_proj[0] - x_circ_proj[0])**2 + (x_bound_proj[1] - x_circ_proj[1])**2)
    h = np.sqrt((x[0] - x_bound_proj[0])**2 + (x[1] - x_bound_proj[1])**2)

    #! Option 1: the bound is perturbed as well. points orthogonal to circle middle stay orthogonal to circle middle
    # if right_left:
    #     bound_perturb = np.array([x_bound_proj[0], x_bound_proj[1] + (1 - np.abs(x_bound_proj[1])/0.16) * omega[2]])
    # else:
    #     bound_perturb = np.array([x_bound_proj[0] + (1 - np.abs(x_bound_proj[0])/0.16) * omega[1], x_bound_proj[1]])
    #! Option 2: the bound is not perturbed
    bound_perturb = x_bound_proj

    circ_perturb = np.array([omega[0] * x_circ_proj[0] + omega[1], omega[0] * x_circ_proj[1] + omega[2]])

    x_pert = h / h_max * circ_perturb + (1 - h / h_max) * bound_perturb

    #! debugging section
    # print(f"x: {x}")
    # print(f"x_pert: {x_pert}")
    # print(f"x_circ_proj: {x_circ_proj}")
    # print(f"x_bound_proj: {x_bound_proj}")
    # print(f"circ_perturb: {circ_perturb}")
    # print(f"bound_perturb: {bound_perturb}")
    # print(f"h / h_max: {h / h_max}")
    #! debugging section

    return np.array([0.16, 0.16]) + x_pert # , np.array([0.16, 0.16]) + x_circ_proj, np.array([0.16, 0.16]) + x_bound_proj

def perturb_mesh(mesh: fe.Mesh, omega: np.array, r: float) -> fe.Mesh:
    perturbed_mesh = fe.Mesh(mesh)
    
    coordinates = mesh.coordinates()
    
    perturbed_coordinates = np.zeros_like(coordinates)
    for i in range(coordinates.shape[0]):
        perturbed_point_coords = perturbation_function(coordinates[i], omega, r)
        perturbed_coordinates[i] = perturbed_point_coords
    
    perturbed_mesh.coordinates()[:] = perturbed_coordinates

    return perturbed_mesh



# #! TEST PERTURBATION FUNCTION
# omega = np.array([1.0, -0.03, 0.02])
# r = 0.02
# x = np.array([0.3, 0.1])
# mesh_resolution = 5

# import mshr 
# bottom_left_corner = fe.Point(0, 0)
# top_right_corner = fe.Point(0.32, 0.32)
# domain = mshr.Rectangle(bottom_left_corner, top_right_corner)
# circ_center = fe.Point((top_right_corner[0] + bottom_left_corner[0])/2, (top_right_corner[1] + bottom_left_corner[1])/2)
# circ_radius = r
# domain = domain - mshr.Circle(circ_center, circ_radius)
# mesh = mshr.generate_mesh(domain, mesh_resolution)

# x_pert, x_circ_proj, x_bound_proj = perturbation_function(x, omega, r)




# import matplotlib.pyplot as plt

# # Plot the mesh and boundary points
# fe.plot(mesh, title='Reference Mesh')

# # Plot the circle
# circle = mshr.Circle(circ_center, circ_radius)
# circle_mesh = mshr.generate_mesh(circle, 64)
# boundary_mesh = fe.BoundaryMesh(circle_mesh, "exterior", order=True)
# boundary_points = boundary_mesh.coordinates()
# plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='blue', s=1, label='Circle Boundary')

# # Plot the perturbed circle
# circle_perturbed = mshr.Circle(fe.Point(0.16 + omega[1], 0.16 + omega[2]), r * omega[0])
# circle_mesh_perturbed = mshr.generate_mesh(circle_perturbed, 64)
# boundary_mesh_perturbed = fe.BoundaryMesh(circle_mesh_perturbed, "exterior", order=True)
# boundary_points_perturbed = boundary_mesh_perturbed.coordinates()
# plt.scatter(boundary_points_perturbed[:, 0], boundary_points_perturbed[:, 1], color='blue', s=1, label='Circle Boundary Perturbed')

# plt.scatter(x[0], x[1], color='black', s=10, label='Original Point')
# plt.scatter(x_circ_proj[0], x_circ_proj[1], color='green', s=10, label='Projected Point on Circle')
# plt.scatter(x_bound_proj[0], x_bound_proj[1], color='cyan', s=10, label='Projected Point on Bound')
# plt.scatter(x_pert[0], x_pert[1], color='red', s=10, label='Perturbed point')
# plt.title('Mesh')
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.legend(loc='upper right')
# plt.xlim(bottom_left_corner[0] - 0.02, top_right_corner[0] + 0.02)
# plt.ylim(bottom_left_corner[1] - 0.02, top_right_corner[1] + 0.02)

# plt.show()



class RandomFieldVExpression(fe.UserExpression):
    def __init__(self, r: float, omega: np.array, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.omega = omega
        self.domain = domain

    def eval(self, values, x):
        perturbed_point = perturbation_function(x=x, omega=self.omega, r=self.r)
        values[0] = perturbed_point[0]
        values[1] = perturbed_point[1]

    def value_shape(self):
        return (2,)
    
class J_minus_TExpression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        J_inv_T = np.linalg.inv(J).T
        values[0] = J_inv_T[0, 0]
        values[1] = J_inv_T[0, 1]
        values[2] = J_inv_T[1, 0]
        values[3] = J_inv_T[1, 1]

    def value_shape(self):
        return (2, 2)
    
class J_helper1Expression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = J[1, 1]
        values[1] = - J[1, 0]

    def value_shape(self):
        return (2, )

class J_helper2Expression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J = self.jacobianProj(x).reshape((2, 2))
        values[0] = - J[0, 1]
        values[1] = J[1, 1]

    def value_shape(self):
        return (2, )
    
class J_determinantExpression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J_det = np.linalg.det(self.jacobianProj(x).reshape((2, 2)))
        if J_det == 0:
            print("Determinant 0.")
            values[0] = 1
        else:
            values[0] = J_det

    def value_shape(self):
        return ()
    
class J_inv_determinantExpression(fe.UserExpression):
    def __init__(self, jacobianProj: fe.Function, domain: fe.Mesh, **kwargs):
        super().__init__(**kwargs)
        self.jacobianProj = jacobianProj
        self.domain = domain

    def eval(self, values, x):
        J_det = np.linalg.det(self.jacobianProj(x).reshape((2, 2)))
        if J_det == 0:
            print("Determinant 0.")
            values[0] = 1
        else:
            values[0] = 1 / J_det

    def value_shape(self):
        return ()
    


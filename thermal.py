from fenics import *
from fenics_shells import *
import numpy as np
from ufl import RestrictedElement
import mshr

from fenics_shells.common.laminates import NM_T

parameters["form_compiler"]["quadrature_degree"] = 2

def normal(y):
    n = cross(y.dx(0), y.dx(1))
    return n/sqrt(inner(n,n))

class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


mesh = RectangleMesh(Point((0, 0)), Point(10, 10), 20, 20)

h = 0.1
thetas = [np.deg2rad(0), np.deg2rad(90)]
E1 = 5000
E2 = 1000
G12 = 500
nu12 = 0.3
G23 = 500
a1 = [1e-6, 1e-6]
a2 = [1e-4, 1e-4]
T_ini = 0.0
T_final = 100

n_layers = len(thetas)
hs = h*np.ones(n_layers)/n_layers
A, B, D = laminates.ABD(E1, E2, G12, nu12, hs, thetas)
Fs = laminates.F(G12, G23, hs, thetas)

element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

U = FunctionSpace(mesh, element)
u, u_t, u_ = TrialFunction(U), TestFunction(U), Function(U)
v_, beta_, w_, R_gamma_, p_ = split(u_)
z_ = as_vector([v_[0], v_[1], w_])
d = as_vector([sin(beta_[1])*cos(beta_[0]), -sin(beta_[0]), cos(beta_[1])*cos(beta_[0])])
F = grad(z_) + as_tensor([[1.0, 0.0],
                          [0.0, 1.0],
                          [Constant(0.0), Constant(0.0)]])

# Riemannian metric (First fundamental form)
class Metric_Tensor(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)
    def eval(self, values, x):
        values[0] = 1.
        values[1] = 0.
        values[2] = 0.
        values[3] = 1.
    def value_shape(self):
        return (2, 2)

# Curvature (Shape tensor; Second fundamental form)
class Curvature_Tensor(UserExpression):
    def init(self, **kwargs):
        super().init(**kwargs)
    def eval(self, values, x):
        values[0] = 0.
        values[1] = 0.
        values[2] = 0.
        values[3] = 0.
    def value_shape(self):
        return (2, 2)

a0 = Metric_Tensor()
b0 = Curvature_Tensor()

# Thermal
T_now = Expression(('Temp'), Temp=T_ini, degree=0)
N_T, M_T = NM_T(E1, E2, G12, nu12, hs, thetas, DeltaT_0=T_now, DeltaT_1=0.0, alpha1=a1, alpha2=a2)

e = 0.5*(F.T*F - a0)
ev = strain_to_voigt(e)
Ai = project(A, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_N = .5*dot(Ai*ev, ev) + .5*dot(N_T, ev)


Fi = project(Fs, TensorFunctionSpace(mesh, 'CG', 1, shape=(2,2)))
psi_T = .5*dot(Fi*R_gamma_, R_gamma_)

k = 0.5*(F.T*grad(d) + grad(d).T*F) - b0
kv = strain_to_voigt(k)
Di = project(D, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_M = .5*dot(Di*kv, kv) + .5*dot(M_T, kv)

Bi = project(B, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_MN = dot(Bi*kv, ev)

h_max = mesh.hmax()
def left(x,on_boundary):
    return x[0] < 1e-10 and on_boundary 

bc_v = DirichletBC(U.sub(0), Constant((0.0,0.0)), left)
bc_R = DirichletBC(U.sub(1), Constant((0.0,0.0)), left)
bc_w = DirichletBC(U.sub(2), Constant(0.0), left)
bcs = [bc_v, bc_R, bc_w]

gamma = F.T*d
L_R = inner_e(gamma - R_gamma_, p_)

L = (psi_M + psi_T + psi_N + psi_MN)*dx + L_R
F = derivative(L, u_, u_t)
J = derivative(F, u_, u)

problem = Problem(J, F, bcs=bcs)
solver = NewtonSolver()

solver.parameters['error_on_nonconvergence'] = False
solver.parameters['maximum_iterations'] = 20
solver.parameters['linear_solver'] = "mumps"
solver.parameters['absolute_tolerance'] = 1E-10
solver.parameters['relative_tolerance'] = 1E-6

t = T_ini
dt = 1
file_z = XDMFFile("output/disp.xdmf")
file_k = XDMFFile("output/curve.xdmf")
while t < T_final:
    T_now.Temp = t
    solver.solve(problem, u_.vector())
    v_h, beta_h, w_h, Rgamma_h, p_h = u_.split(deepcopy=True)
    z_h = project(z_, VectorFunctionSpace(mesh, "CG", 1, dim=3))
    n = normal(z_h)
    a0 = grad(z_h).T*grad(z_h)
    a = project(a0, TensorFunctionSpace(mesh, 'CG', 1, shape=(2,2)))
    b0 = -0.5*(grad(z_h).T*grad(n)+grad(n).T*grad(z_h))
    b = project(b0, TensorFunctionSpace(mesh, 'CG', 1, shape=(2,2)))
    b.rename('z', 'z')
    file_z.write(b, t)
    t +=dt

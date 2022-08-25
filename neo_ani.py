from ast import Index
from fenics import *
from importlib_metadata import metadata
from fenics_shells import *
import numpy as np
from ufl import RestrictedElement
import mshr
from mpi4py import MPI
comm = MPI.COMM_WORLD

hs = [0.45*1e-3, 0.05*1e-3]

class Passive(UserExpression):
    def eval(self, val, x):
        a = np.deg2rad(80)
        b = np.deg2rad(0)
        L = 0.015
        val[0] = -(b-a)/L*x[1] + a

def z_coordinates(hs):
    r"""Return a list with the thickness coordinate of the top surface of each layer
    taking the midplane as z = 0.

    Args:
        hs: a list giving the thinckesses of each layer
            ordered from bottom (layer - 0) to top (layer n-1).

    Returns:
        z: a list of coordinate of the top surface of each layer
           ordered from bottom (layer - 0) to top (layer n-1)
    """

    z0 = sum(hs)/2.
    #z = [(sum(hs)/2.- sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)];
    z = [(-sum(hs)/2. + sum(hs for hs in hs[0:i])) for i in range(len(hs)+1)]
    return z

#mesh = RectangleMesh(comm, Point((0, 0)), Point(L, W), 40, 4)
mesh = Mesh(comm, '/workspace/mesh/wing.xml')

parameters["form_compiler"]["quadrature_degree"] = 2

L = 100*1e-3
W = 10*1e-3
h = 0.5*1e-3
nu = 0.49
a1 = 0.7
a2 = 3.25
a3 = -3.7
mu1 = 54.88*1e3
mu2 = 910
mu3 = -6.3
mu = 0.5*(a1*mu1+a2*mu2+a3*mu3)
epsr = 4.7
eps0 = 8.85*1e-12
V_init = 0.0
dV = 1
V_final = 2000

E1 = 100*1e3
E2 = 1e-3
G12 = 1e-3
nu12 = 0.3
theta = Function(FunctionSpace(mesh, 'CG', 1))
theta.interpolate(Passive())

z = z_coordinates(hs)

# Capacity
c = 2*epsr*eps0/h
# Coupling factor
alpha = h/4

# Membrane stiffness
Y = 3*mu
# A = Y/(1-nu**2)*(z[1]-z[0])
A = Y/(1-nu**2)*h
# Bending stiffness
#D = Y/12/(1-nu**2)*(z[1]**3-z[0]**3)/3
D = Y/12/(1-nu**2)*h**3/3

element = MixedElement([VectorElement("Lagrange", triangle, 1, dim=3),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge")])

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

phi0_expression = Expression(['x[0]', 'x[1]', 'phi0_z'], phi0_z=0., degree=4)
phi0 = project(phi0_expression, FunctionSpace(mesh, VectorElement("Lagrange", triangle, degree = 1, dim = 3)))
beta0_expression = Expression(('beta0_x', 'beta0_y'), beta0_x=0., beta0_y=0., beta0_z=0., degree=4)
beta0 = project(beta0_expression, FunctionSpace(mesh, VectorElement("CG", triangle, degree = 2, dim = 2)))

def director(beta):
    return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])

d0 = director(beta0)

Q = FunctionSpace(mesh, element)
q_, q, q_t = Function(Q), TrialFunction(Q), TestFunction(Q)
u_, beta_, rg_, p_ = split(q_)

def left(x,on_boundary):
    return x[0] < 1e-10 and on_boundary
def right(x,on_boundary):
    return x[0] > L-1e-10 and on_boundary
bc_u = DirichletBC(Q.sub(0), Constant((0.0,0.0,0.0)), left)
bc_R = DirichletBC(Q.sub(1), Constant((0.0,0.0)), left)
bcs = [bc_u, bc_R]

# Deformation gradient
F = grad(u_) + grad(phi0)
# Director
d = director(beta_ + beta0)
# First metric
a0 = grad(phi0).T*grad(phi0)
# Second metric
b0 = -0.5*(grad(phi0).T*grad(d0) + grad(d0).T*grad(phi0))
# Strain measures
e = lambda F: 0.5*(F.T*F - a0)
k = lambda F, d: -0.5*(F.T*grad(d)+grad(d).T*F) - b0
gamma = lambda F, d: F.T*d - grad(phi0).T*d0

def Wm(F):
    C = F.T*F
    return 0.5*A/4*(tr(C) + 1/det(C) - 3)

def Wb(F, d):
    C = F.T*F
    K = inv(C)*k(F, d)
    return 0.5/det(C)*D*((tr(K)**2-det(K)))

V = Expression('Voltage', Voltage=V_init, degree=0)

def Wmel(F):
    C = F.T*F
    #return -0.5*c*V**2*det(C)
    #return -c*V*det(C)*dV
    return -8*eps0*epsr*V/(h**2)*(z[1]-z[0])*det(C)*dV

def Wbel(F, d):
    C = F.T*F
    K = inv(C)*k(F, d)
    #return -2*alpha*c*V*det(C)*tr(K)*dV
    return -4*eps0*epsr*V/(h**2)*(z[1]**2-z[0]**2)*det(C)*tr(K)*dV

def Fs(rg_):
    return  0.5*h*mu*inner(rg_, rg_)

def ABD(E1, E2, G12, nu12, theta):
    Ap = 0.*Identity(3)
    Bp = 0.*Identity(3)
    Dp = 0.*Identity(3)
    Qbar = rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta)
    Ap += Qbar*(z[2]-z[1])
    Bp += .5*Qbar*(z[2]**2-z[1]**2)
    Dp += 1./3.*Qbar*(z[2]**3-z[1]**3)
    return (Ap, Bp, Dp)

def Fsp(G13, G23, theta):
    Fp = 0.*Identity(2)
    Q_shear_theta = rotated_lamina_stiffness_shear(G13, G23, theta)
    Fp += Q_shear_theta*(z[2]-z[1])
    return Fp

Ap, Bp, Dp = ABD(E1, E2, G12, nu12, theta)
Fsp = Fsp(G12, G12, theta)
ev = strain_to_voigt(e(F))
Ai = project(Ap, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_N = .5*dot(Ai*ev, ev)
Fi = project(Fsp, TensorFunctionSpace(mesh, 'CG', 1, shape=(2,2)))
psi_T = .5*dot(Fi*rg_, rg_)
kv = strain_to_voigt(k(F, d))
Di = project(Dp, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_M = .5*dot(Di*kv, kv) 
Bi = project(Bp, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
psi_MN = dot(Bi*kv, ev)

Pi_aniso = psi_M + psi_MN + psi_N + psi_T
Pi = Wb(F, d) + Wbel(F, d) + Wm(F) + Wmel(F) + Fs(rg_) + Pi_aniso
L_R = inner_e(gamma(F, d) - rg_, p_) 

Pi = Pi*dx + L_R
dPi = derivative(Pi, q_, q_t)
J = derivative(dPi, q_, q)

problem = Problem(J, dPi, bcs=bcs)
solver = NewtonSolver()

solver.parameters['error_on_nonconvergence'] = False
solver.parameters['maximum_iterations'] = 20
solver.parameters['linear_solver'] = "mumps"
solver.parameters['absolute_tolerance'] = 1E-10
solver.parameters['relative_tolerance'] = 1E-6

V_cur = V_init
file_phi = XDMFFile("output/disp.xdmf")
file_angle = XDMFFile("output/angle.xdmf")
angle = project(as_vector((cos(theta), sin(theta))), VectorFunctionSpace(mesh, 'CG', 1))
file_angle.write(angle)
disp0_expression = Expression(['0', '0', '0'], degree=4)
disp0 = Function(VectorFunctionSpace(mesh, 'CG', 1, dim=3))
disp0 = project(disp0_expression, FunctionSpace(mesh, VectorElement("Lagrange", triangle, degree = 1, dim = 3)))
while V_cur < V_final:
    V.Voltage = V_cur
    solver.solve(problem, q_.vector())
    u_h, beta_h, _, _ = q_.split(deepcopy=True)
    disp = project(disp0+u_h, VectorFunctionSpace(mesh, 'CG', 1, dim=3))
    assign(phi0, project(phi0+u_h, VectorFunctionSpace(mesh, 'CG', 1, dim=3)))
    assign(beta0, project(beta0+beta_h, VectorFunctionSpace(mesh, 'CG', 2, dim=2)))
    assign(disp0, project(disp0+u_h, VectorFunctionSpace(mesh, 'CG', 1, dim=3)))
    disp.rename('disp', 'disp')
    file_phi.write(disp, V_cur)
    V_cur += dV
    if V_cur % 100 == 0:
        file = XDMFFile('output/frame/step_{}.xdmf'.format(int(V_cur)))
        file.write(disp)


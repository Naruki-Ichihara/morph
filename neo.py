from ast import Index
from fenics import *
from importlib_metadata import metadata
from fenics_shells import *
import numpy as np
from ufl import RestrictedElement
import mshr
from mpi4py import MPI
comm = MPI.COMM_WORLD


parameters["form_compiler"]["quadrature_degree"] = 2

L = 100*1e-3
W = 50*1e-3
h = 1.0*1e-3
nu = 0.5
mu = 20698
epsr = 4.7
eps0 = 8.85*1e-12
V_init = 0.0
dV = 1
V_final = 4500

# Capacity
c = 2*epsr*eps0/h
# Coupling factor
alpha = h/4

# Membrane stiffness
Y = 3*mu
A = Y*h/(1-nu**2)

# Bending stiffness
D = Y*h**3/12/(1-nu**2)

mesh = RectangleMesh(comm, Point((0, 0)), Point(L, W), 20, 10)

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
    return x[0] < 1e-2 and on_boundary
def right(x,on_boundary):
    return x[0] > L-1e-2 and on_boundary
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
    return -c*V*det(C)*dV

def Wbel(F, d):
    C = F.T*F
    K = inv(C)*k(F, d)
    return -2*alpha*c*V*det(C)*tr(K)*dV

def Fs(rg_):
    return  0.5*h*mu*inner(rg_, rg_)

Pi = Wb(F, d) + Wbel(F, d) + Wm(F) + Wmel(F) + Fs(rg_)
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


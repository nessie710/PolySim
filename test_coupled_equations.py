from mpi4py import MPI
from dolfinx import mesh 
from dolfinx.mesh import locate_entities, meshtags 
from dolfinx.fem import FunctionSpace
from dolfinx import fem 
import numpy as np 
import ufl 
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc 
from dolfinx import io
import ufl.finiteelement 
import dolfinx.fem.petsc
import basix.ufl

# Creating an interval as mesh
h = 0.1
x_min, x_max = -2,2
nx = int((x_max - x_min)/h)
domain = mesh.create_interval(MPI.COMM_WORLD, nx, points = (x_min, x_max))
x = ufl.SpatialCoordinate(domain) # Coordinates of domain used for defining general functions

element_family = basix.ElementFamily.P
element_degree = 1
variant = basix.LagrangeVariant.equispaced

# Define the element using Basix

el1 = basix.ufl.element(element_family, domain.topology.cell_name(), element_degree, variant)
el2 = basix.ufl.element(element_family, domain.topology.cell_name(), element_degree, variant)
mel = basix.ufl.mixed_element(([el1, el2]))
Z = fem.functionspace(domain, mel)

V = fem.functionspace(domain, ("CG", 1)) 
U = fem.functionspace(domain, ("DG", 1))

(w_trial, u_trial) = ufl.TrialFunctions(Z)
(w_test, u_test) = ufl.TestFunctions(Z)
Z_sol = fem.Function(Z)

# Constants
m  = 2
alpha_par = 1/3
beta_par = 1/3
kappa_par = 1/12
C = 1/32
t_start  = 0.3
dt = 0.1

def Phi(u):
    return u**m 

def exact_sol(x,t):
    return ufl.operators.max_value(0,1/(t**alpha_par) * (C - kappa_par * (ufl.operators.sqrt(x[0]**2)/(t**beta_par))**2 )**(1/(m-1)))

initial_u = exact_sol(x,t_start)
expr_u = fem.Expression(initial_u, U.element.interpolation_points())

# Defining the function spaces for all the functions
u_n = fem.Function(U)
u_n.name = "u_n"

u_n_i = fem.Function(U)
u_n_i.name = "u_n_i"

w_n = fem.Function(V)
w_n.name = "w_n"

w_n_i = fem.Function(V)
w_n_i.name = "w_n_i"

# Defining the solution variables
uh = fem.Function(U)
uh.name = "uh"

wh = fem.Function(V)
wh.name = "wh"

initial_u = exact_sol(x,t_start)
expr_u = fem.Expression(initial_u, U.element.interpolation_points())
expr_w = fem.Expression(Phi(initial_u), V.element.interpolation_points())

u_n.interpolate(expr_u)
u_n_i.interpolate(expr_u)
w_n.interpolate(expr_w)
w_n_i.interpolate(expr_w)

L_par = 1
L_const = fem.Constant(domain, ScalarType(L_par))

a = (u_trial * w_test + dt * ufl.inner(ufl.grad(w_trial), ufl.grad(w_test)) + L_const * u_trial * u_test - w_trial * u_test) * ufl.dx
L = (u_n * w_test + L_const * u_n_i * u_test - Phi(u_n_i) * u_test) * ufl.dx
  
bilinear_form = fem.form(a)
linear_form = fem.form(L)
  
b = dolfinx.fem.petsc.assemble_vector(linear_form)
b.assemble()
A = dolfinx.fem.petsc.assemble_matrix(bilinear_form)
A.assemble()
  
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY) # PREONLY means you only use the preconditioner
#solver.setTolerances(rtol=1e-12, atol = 1e-60)
solver.getPC().setType(PETSc.PC.Type.LU) # The LU preconditioner just solves the system using LU factorization.
  
solver.setOperators(A)
  
solver.solve(b, Z_sol.vector)
(wh, uh) = Z_sol.split()
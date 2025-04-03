#%% Import statements

from mpi4py import MPI
import ipyparallel as ipp

from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector

from dolfinx.io import gmshio, XDMFFile
from dolfinx import mesh, fem
from dolfinx.fem import functionspace
import numpy as np
import ufl
from dolfinx import default_scalar_type, default_real_type
from dolfinx import geometry as gm
from dolfinx.fem.petsc import LinearProblem
import pyvista
from dolfinx import plot
import ufl.finiteelement
try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)

from simple_mesh_elements import gmsh_simple_1_electrode_domain
from create_mesh_function import create_mesh
import matplotlib.pyplot as plt
import matplotlib
import dolfinx.fem.petsc
import basix.ufl
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
print(PETSc.ScalarType)
# assert np.dtype(PETSc.ScalarType).kind == 'c'
from boundary_conditions import assemble_boundary_conditions_stationary, assemble_boundary_conditions_AC
from PNP_equation_sets import assemble_stationary_problem, assemble_AC_problem
from extract_cutlines import extract_central_cutline_1D
import ufl
from dolfinx import default_real_type
from dolfinx.fem import Function, dirichletbc, form, functionspace, locate_dofs_geometrical
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, derivative, dx, grad, inner

from dolfinx.fem.petsc import (assemble_matrix, assemble_vector,
                               create_matrix)

def mpi_print(s):
    print(f"Rank {MPI.COMM_WORLD.rank}: {s}", flush=True)



# Constants

epsilon0 = 8.8541878128e-12
epsilon_r_w = 80
epsilon_w = epsilon_r_w*epsilon0
q = 1.60217e-19
NA = 6.02214076e23
k = 1.380649e-23
T = 300
D1 = 1e-9
D2 = 1e-9

d = 0.72e-9

phi_char = k*T/q
c_char = 1/(NA*d**3)
x_char = np.sqrt(epsilon_w*k*T/((q**2)*NA*c_char))  
J1_char = q*D1*c_char*NA/x_char
J2_char = q*D2*c_char*NA/x_char
t_char = x_char**2/D1
f_char = 1/t_char

c_bulk = 170
c_bulk_scaled = c_bulk/c_char
Vapp = 0.1
Vapp_scaled = Vapp/phi_char
V_bulk = 0
V_bulk_scaled = V_bulk/phi_char
L = 1e-8
L_scaled = L/x_char
print(c_bulk_scaled)


#%%px
try:
    from dolfinx.graph import partitioner_scotch
    has_scotch = True
except ImportError:
    has_scotch = False
try:
    from dolfinx.graph import partitioner_kahip
    has_kahip = True
except ImportError:
    has_kahip = False
try:
    from dolfinx.graph import partitioner_parmetis
    has_parmetis = True
except ImportError:
    has_parmetis = False
    
# Define Mesh

domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(0.0, L_scaled), nx=500000)
topology, geometry = domain.topology, domain.geometry
eps = ufl.Constant(domain, np.finfo(float).eps)
cluster = ipp.Cluster(engines="mpi", n=1)
rc = cluster.start_and_connect_sync()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Create connectivity
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
x = ufl.SpatialCoordinate(domain)


element_family_vector = basix.ElementFamily.BDM
element_degree = 1
variant = basix.LagrangeVariant.equispaced
vector_el = basix.ufl.element(element_family_vector, domain.topology.cell_name(), element_degree, variant)
#vector_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), element_degree)


element_family_scalar = basix.ElementFamily.P
element_degree = 2
variant = basix.LagrangeVariant.equispaced
scalar_el = basix.ufl.element(element_family_scalar, domain.topology.cell_name(), element_degree, variant)

mel = basix.ufl.mixed_element([scalar_el, scalar_el, scalar_el])

V = fem.functionspace(domain, mel)

v1, v2, vphi= ufl.TestFunctions(V)
u = fem.Function(V)
c1, c2, phi= ufl.split(u)

def c0_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = c_bulk_scaled
    return values


def V0_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 0
    return values

u.sub(0).interpolate(c0_init)
u.sub(1).interpolate(c0_init)
u.sub(2).interpolate(V0_init)

n = ufl.FacetNormal(domain)

# IMPORT STATIONARY EQUATION SET

F1 = -ufl.inner(ufl.grad(c1)[0], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[0], v1) * ufl.dx - ufl.inner((c1/(1-c1-c2))*(ufl.grad(c1)[0]+ ufl.grad(c2)[0]), v1)*ufl.dx 
F2 = -ufl.inner(ufl.grad(c2)[0], v2) * ufl.dx + ufl.inner(c2 * ufl.grad(phi)[0], v2) * ufl.dx - ufl.inner((c2/(1-c1-c2))*(ufl.grad(c1)[0]+ ufl.grad(c2)[0]), v2)*ufl.dx

F7 = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - ufl.inner((c1-c2), vphi) * ufl.dx
F8 = ufl.inner(-ufl.grad(c1) - c1 * ufl.grad(phi) - (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v1)) * ufl.dx
F9 = ufl.inner(-ufl.grad(c2) + c2 * ufl.grad(phi) - (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v2)) * ufl.dx

#F7_normal = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - ufl.inner((NA*q/epsilon_w)*(c1-c2), vphi) * ufl.dx
#F8_normal = ufl.inner(-D1*ufl.grad(c1) - (D1/(k*T/q))*c1 * ufl.grad(phi) - (NA*d**3*c1/(1-NA*d**3*c1-NA*d**3*c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v1)) * ufl.dx
#F9_normal = ufl.inner(-D2*ufl.grad(c1) + (D2/(k*T/q))*c2 * ufl.grad(phi) - (NA*d**3*c2/(1-NA*d**3*c1-NA*d**3*c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v2)) * ufl.dx


# hk = ufl.CellDiameter(domain)
# b = - ufl.grad(D1/x_char * phi)

# def psi(q):
#     return ufl.conditional(q > 1, 1, q)

# nb = ufl.sqrt(ufl.dot(b,b))

# Pe = 0.33*nb*hk/(2*D1)

# sigma = hk/(2*nb)*psi(Pe)
# v1_supg = ufl.inner(sigma*b, ufl.grad(v1))
# v2_supg = ufl.inner(sigma*b, ufl.grad(v2))

# supg1 = ufl.inner(ufl.div(ufl.grad(c1) + c1 * ufl.grad(phi) + (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))), v1_supg) * ufl.dx
# supg2 = ufl.inner(ufl.div(ufl.grad(c2) - c2 * ufl.grad(phi) + (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))), v2_supg) * ufl.dx

# b      = (1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))
# nb     = ufl.sqrt(ufl.dot(b,b))
# tau = 1/ufl.sqrt((2*nb/hk)**2 + (4*D1/(hk**2))**2)
# L1 = 0.5*(v1*ufl.div(ufl.grad(phi) + (ufl.grad(c1)+ufl.grad(c2))/(1-c1-c2)) + 2*ufl.inner(ufl.grad(phi) + (ufl.grad(c1)+ufl.grad(c2))/(1-c1-c2),ufl.grad(v1)))
# L2 = 0.5*(v2*ufl.div(ufl.grad(phi) + (ufl.grad(c1)+ufl.grad(c2))/(1-c1-c2)) + 2*ufl.inner(ufl.grad(phi) + (ufl.grad(c1)+ufl.grad(c2))/(1-c1-c2),ufl.grad(v2)))
# supg = ufl.div(-ufl.grad(c1) - c1 * ufl.grad(phi) - (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)))*tau*L1*ufl.dx + ufl.div(-ufl.grad(c2) + c2 * ufl.grad(phi) - (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)))*tau*L2*ufl.dx

F = F7 + F8 + F9 #+ supg1 + supg2

# SET BOUNDARY CONDITIONS 
def boundary_R(x):
    return np.isclose(x[0], L_scaled)

def boundary_L(x):
    return np.isclose(x[0], 0)

V_split = V.sub(2)
V_potential, _ = V_split.collapse()
ud = fem.Function(V_potential)
ud.interpolate(lambda x : x[0]*0 + V_bulk_scaled)
dofs_bulk = fem.locate_dofs_geometrical((V_split,V_potential), boundary_R)
bc_potential_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)

ud = fem.Function(V_potential)
ud.interpolate(lambda x : x[0]*0 + Vapp_scaled)
dofs_surface = fem.locate_dofs_geometrical((V_split,V_potential),boundary_L)
bc_potential_surface = fem.dirichletbc(ud, dofs_surface, V_split)


V_split = V.sub(0)
V_c1, _ = V_split.collapse()
ud = fem.Function(V_c1)
ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled)
dofs_bulk = fem.locate_dofs_geometrical((V_split,V_c1), boundary_R)
bc_c1_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)


V_split = V.sub(1)
V_c2, _ = V_split.collapse()
ud = fem.Function(V_c2)
ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled)
dofs_bulk = fem.locate_dofs_geometrical((V_split,V_c2), boundary_R)
bc_c2_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)  

bcs = [bc_potential_bulk, bc_potential_surface, bc_c1_bulk, bc_c2_bulk]



# SET PROBLEM


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bc], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bc, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bc)
        J.assemble()


# Create nonlinear problem
problem = NonlinearPDE_SNESProblem(F, u, bcs)

b = dolfinx.la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J =fem.petsc.create_matrix(problem.a)

# Create SNES solver and solve
snes = PETSc.SNES().create()
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-9, max_it=200)
snes.getKSP().setType("gmres")
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.getKSP().getPC().setType("lu")

# For SNES line search to function correctly it is necessary that the
# u.x.petsc_vec in the Jacobian and residual is *not* passed to
# snes.solve.
x = u.x.petsc_vec.copy()
x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dolfinx.log.set_log_level(dolfinx.cpp.log.LogLevel.INFO)

snes.solve(None, x)
assert snes.getConvergedReason() > 0






# problem = NonlinearProblem(F, u, bcs = bcs)
# solver = NewtonSolver(MPI.COMM_WORLD, problem)
# solver.convergence_criterion = "incremental"
# solver.rtol = np.sqrt(np.finfo(default_real_type).eps)*1e-2
# solver.max_it = 200

# ksp = solver.krylov_solver
# opts = PETSc.Options()  
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "fgmres"
# opts[f"{option_prefix}pc_type"] = "lu"
# sys = PETSc.Sys()  
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()

# dolfinx.log.set_log_level(dolfinx.cpp.log.LogLevel.INFO)



# # SOLVE PROBLEM AND PROPAGATE TO GHOSTS
# solver.solve(u)
u.x.scatter_forward()


points_on_proc, cells = extract_central_cutline_1D(domain, L_scaled)


points_gathered = MPI.COMM_WORLD.gather(points_on_proc, root=0)
cells_gathered = MPI.COMM_WORLD.gather(cells, root=0)

c1_local = u.sub(0).eval(points_on_proc, cells)*c_char
c2_local = u.sub(1).eval(points_on_proc, cells)*c_char
phi_local = u.sub(2).eval(points_on_proc, cells)*phi_char



J1 = -ufl.grad(c1) - c1 * ufl.grad(phi) - (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))
V_vector_temp = fem.functionspace(domain, vector_el)
J1_func = fem.Function(V_vector_temp)
#J1_func = fem.Function(V)
expr_J1 = fem.Expression(J1, V_vector_temp.element.interpolation_points())
J1_func.interpolate(expr_J1)
form = fem.form(J1_func)
integral_j1_value = fem.assemble_scalar(form)
j1_local = integral_j1_value*J1_char*(x_char**2)

J2 = -ufl.grad(c2) + c2 * ufl.grad(phi) - (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))
J2_func = fem.Function(V_vector_temp)
#J2_func = fem.Function(V)
expr_J2 = fem.Expression(J2, V_vector_temp.element.interpolation_points())
J2_func.interpolate(expr_J2)
form = fem.form(J2_func)
integral_j2_value = fem.assemble_scalar(form)
j2_local = integral_j2_value *J2_char*(x_char**2)




c1_gathered = comm.gather(c1_local,root=0)
c2_gathered = comm.gather(c2_local,root=0)
phi_gathered = comm.gather(phi_local,root=0)
j1_gathered = comm.gather(j1_local, root=0)
j2_gathered = comm.gather(j2_local, root=0)


if MPI.COMM_WORLD.rank == 0:  

    points_gathered_filtered = [p for p in points_gathered if p.size > 0]
    if len(points_gathered_filtered) > 0:
        points_combined = np.vstack(points_gathered_filtered)  
    else:
        points_combined = points_gathered_filtered
    c1_combined = np.vstack(c1_gathered)
    c2_combined = np.vstack(c2_gathered)
    phi_combined = np.vstack(phi_gathered)
    j1_combined = np.vstack(j1_gathered)
    j2_combined = np.vstack(j2_gathered)

    # print(np.shape(points_combined))
    sort_indices = np.argsort(points_combined[:, 0])  
    points_combined = points_combined[sort_indices]
    c1_combined = c1_combined[sort_indices]
    c2_combined = c2_combined[sort_indices]
    phi_combined = phi_combined[sort_indices]
    j1_combined = j1_combined[sort_indices]
    j2_combined = j2_combined[sort_indices]
    




    print("Simulated concentration")
    print(c2_combined[0])
    print("Theoretical concentration")
    print(c_bulk*np.exp(phi_combined[0]/(k*T/q)) / (1- 2*NA*d**3*c_bulk + 2*NA*d**3*c_bulk*np.cosh(phi_combined[0]/(k*T/q))))
    print("PNP concentration")
    print(c_bulk*np.exp(phi_combined[0]/(k*T/q)))
    fig = plt.figure()
    plt.plot(points_combined[:, 0]*x_char, c1_combined, "k", linewidth=2, label="c1")
    plt.plot(points_combined[:, 0]*x_char, c2_combined, "b", linewidth=2, label="c2")
    plt.plot(points_combined[:,0]*x_char, c_bulk*np.exp(phi_combined/(k*T/q)) / (1- 2*NA*d**3*c_bulk + 2*NA*d**3*c_bulk*np.cosh(phi_combined/(k*T/q))), "green", label="c2 theoretical")
    plt.xscale("linear")

    plt.grid(True)
    plt.xlabel("x")
    plt.legend()
    

    fig2 = plt.figure()
    plt.plot(points_combined[:, 0]*x_char, phi_combined, "r", linewidth=2, label="phi")
    plt.grid(True)
    plt.xlabel("x")
    plt.xscale("linear")
    plt.legend()
    print("Done")
    plt.show()







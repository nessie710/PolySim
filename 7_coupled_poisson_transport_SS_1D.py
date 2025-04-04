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
#assert np.dtype(PETSc.ScalarType).kind == 'c'
from boundary_conditions import assemble_boundary_conditions_stationary, assemble_boundary_conditions_AC
from PNP_equation_sets import assemble_stationary_problem, assemble_AC_problem
from extract_cutlines import extract_central_cutline_1D


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
c_bulk = 1

phi_char = k*T/q
c_char = c_bulk
x_char = np.sqrt(epsilon_w*k*T/((q**2)*2*c_char*NA))  
J1_char = q*D1*c_char*NA/x_char
J2_char = q*D2*c_char*NA/x_char
t_char = x_char**2/D1
f_char = 1/t_char

c_bulk_scaled = c_bulk/c_char
Vapp = 1e-3
Vapp_scaled = Vapp/phi_char
V_bulk = 0
V_bulk_scaled = V_bulk/phi_char
Vapp_AC = 1e-3
Vapp_AC_scaled = Vapp_AC/phi_char
L = 1e-7
L_scaled = L/x_char
R_sens = 90e-9
R_sens_scaled = R_sens/x_char
frequencies = np.logspace(3,9,10)/f_char

omegas = 2*np.pi*frequencies
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

domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(0.0, L_scaled), nx=500)
topology, geometry = domain.topology, domain.geometry

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

# element_family_vector = basix.ElementFamily.BDM
# element_degree = 1
# variant = basix.LagrangeVariant.equispaced
# vector_el = basix.ufl.element(element_family_vector, domain.topology.cell_name(), element_degree, variant)

element_family_scalar = basix.ElementFamily.P
element_degree = 2
variant = basix.LagrangeVariant.equispaced
scalar_el = basix.ufl.element(element_family_scalar, domain.topology.cell_name(), element_degree, variant)

mel = basix.ufl.mixed_element([scalar_el, scalar_el, scalar_el])

V = fem.functionspace(domain, mel)

v1, v2, vphi= ufl.TestFunctions(V)
u = fem.Function(V)
c1, c2, phi= ufl.split(u)

n = ufl.FacetNormal(domain)

# IMPORT STATIONARY EQUATION SET
#F1 = -ufl.inner(ufl.grad(c1)[0], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[0], v1) * ufl.dx
#F4 = ufl.inner(ufl.grad(c2)[0], v2) * ufl.dx - ufl.inner(c2 * ufl.grad(phi)[0], v2) * ufl.dx
F7 = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - (1/2)*ufl.inner((c1-c2), vphi) * ufl.dx
F8 = ufl.inner(-ufl.grad(c1) - c1 * ufl.grad(phi), ufl.grad(v1)) * ufl.dx
F9 = ufl.inner(-ufl.grad(c2) + c2 * ufl.grad(phi), ufl.grad(v2)) * ufl.dx

F = F7 + F8 + F9

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


problem = NonlinearProblem(F, u, bcs = bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
solver.max_it = 200

ksp = solver.krylov_solver
opts = PETSc.Options()  
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()  
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

dolfinx.log.set_log_level(dolfinx.cpp.log.LogLevel.INFO)



# SOLVE PROBLEM AND PROPAGATE TO GHOSTS
solver.solve(u)
u.x.scatter_forward()


points_on_proc, cells = extract_central_cutline_1D(domain, L_scaled)


points_gathered = MPI.COMM_WORLD.gather(points_on_proc, root=0)
cells_gathered = MPI.COMM_WORLD.gather(cells, root=0)

c1_local = u.sub(0).eval(points_on_proc, cells)*c_char
c2_local = u.sub(1).eval(points_on_proc, cells)*c_char
phi_local = u.sub(2).eval(points_on_proc, cells)*phi_char



points = np.array([[0.5]], dtype=np.float64)
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
cells_local = range(0,num_cells_local)

x_expr = dolfinx.fem.Expression(ufl.SpatialCoordinate(domain), points)
coords = x_expr.eval(domain, cells_local)
J1 = -ufl.grad(c1) - c1 * ufl.grad(phi)
#V_vector_temp = fem.functionspace(domain, scalar_el)
#J1_func = fem.Function(V_vector_temp)
#J1_func = fem.Function(V)
expr_J1 = fem.Expression(J1, points)
#J1_func.interpolate(expr_J1)
# form = fem.form(J1_func)
# integral_j1_value = fem.assemble_scalar(form)
j1_local = expr_J1.eval(domain,cells_local)*J1_char*(x_char**2)

J2 = -ufl.grad(c2) + c2 * ufl.grad(phi)
#J2_func = fem.Function(V_vector_temp)
#J2_func = fem.Function(V)
expr_J2 = fem.Expression(J2,points)
#J2_func.interpolate(expr_J2)
# form = fem.form(J2_func)
# integral_j2_value = fem.assemble_scalar(form)
j2_local = expr_J2.eval(domain,cells_local) *J2_char*(x_char**2)

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


    print(np.shape(points_combined))
    sort_indices = np.argsort(points_combined[:, 0])  
    points_combined = points_combined[sort_indices]
    c1_combined = c1_combined[sort_indices]
    c2_combined = c2_combined[sort_indices]
    phi_combined = phi_combined[sort_indices]
    
    print(c2_combined[0]/(c_bulk*np.exp(phi_combined[0]/phi_char)))
    fig = plt.figure()
    plt.plot(points_combined[:, 0]*x_char, c1_combined, "k", linewidth=2, label="c1")
    plt.plot(points_combined[:, 0]*x_char, c2_combined, "b", linewidth=2, label="c2")
    
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


    fig3 = plt.figure()
    plt.plot(coords*x_char, j1_combined, "r", linewidth=2, label="j1")
    plt.plot(coords*x_char, j2_combined, "b", linewidth=2, label="j2")
    plt.xlabel("x")
    plt.grid(True)
    plt.xscale("linear")
    plt.yscale("log")
    plt.legend()

    plt.show()




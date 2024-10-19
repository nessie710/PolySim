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
#import pyvista
from dolfinx import plot
import ufl.finiteelement
try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)

#from simple_mesh_elements import gmsh_5x5_cylinder
#from create_mesh_function import create_mesh
import matplotlib.pyplot as plt
import matplotlib
import dolfinx.fem.petsc
import basix.ufl
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'
from boundary_conditions import assemble_boundary_conditions_stationary_electrode_matrix, assemble_boundary_conditions_AC_electrode_matrix
from PNP_equation_sets import assemble_stationary_problem, assemble_AC_problem
from extract_cutlines import extract_central_cutline
import multiphenicsx as mp
import viskex


def mpi_print(s):
    print(f"Rank {MPI.COMM_WORLD.rank}: {s}", flush=True)



# Constants
epsilon0 = 8.8541878128e-12
epsilon_r_w = 80
epsilon_w = epsilon_r_w*epsilon0
epsilon_r_PEDOT = 40
epsilon_PEDOT = epsilon_r_PEDOT*epsilon0
q = 1.60217e-19
NA = 6.02214076e23
k = 1.380649e-23
T = 300
D1 = 1e-9
D2 = 1e-9
c_bulk = 1
R = 2000
Dp = 2e-14
Cv = 59

phi_char = k*T/q
c_char = c_bulk
x_char = np.sqrt(epsilon_w*k*T/((q**2)*2*c_char*NA))
p_char = Cv*phi_char/(q*NA)  
xp_char = np.sqrt(epsilon_w*k*T/((q**2)*p_char*NA))
J1_char = q*D1*c_char*NA/x_char
J2_char = q*D2*c_char*NA/x_char
Jp_char = q*Dp*p_char*NA/xp_char
t_char = x_char**2/D1
f_char = 1/t_char

c_bulk_scaled = c_bulk/c_char
Vapp = 0
Vapp_scaled = Vapp/phi_char
V_bulk = 0
V_bulk_scaled = V_bulk/phi_char

L = 20e-6
H_CP = 650e-9

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
r = 3
mesh_size = 1. / 4.
# Load mesh
gmsh.initialize()
gmsh.model.add("mesh")
p0 = gmsh.model.geo.addPoint(0,0,0, mesh_size)
p1 = gmsh.model.geo.addPoint(0, H_CP, 0, mesh_size)
p2 = gmsh.model.geo.addPoint(0, L, 0, mesh_size)
p3 = gmsh.model.geo.addPoint(L, 0, 0, mesh_size)
p4 = gmsh.model.geo.addPoint(L, H_CP, 0, mesh_size)
p5 = gmsh.model.geo.addPoint(L, L, 0, mesh_size)
c0 = gmsh.model.geo.addLine(p1,p0)
c1 = gmsh.model.geo.addLine(p1,p2)
c2 = gmsh.model.geo.addLine(p2,p5)
c3 = gmsh.model.geo.addLine(p3,p4)
c4 = gmsh.model.geo.addLine(p5,p4)
c5 = gmsh.model.geo.addLine(p4,p1)
c6 = gmsh.model.geo.addLine(p0,p3)

boundary = gmsh.model.geo.addCurveLoop([c1,c2,c4,c5])
interface = gmsh.model.geo.addCurveLoop([c0,c6,c3,c5])
domain = gmsh.model.geo.addPlaneSurface([boundary])
polymer = gmsh.model.geo.addPlaneSurface([interface])
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, [c1,c2,c4,c5], 2)
gmsh.model.addPhysicalGroup(1, [c0,c6,c3,c5], 1)
gmsh.model.addPhysicalGroup(2, [boundary], 0)
gmsh.model.addPhysicalGroup(2, [interface], 3)
gmsh.model.mesh.generate(2)

# bulk_marker = 11
# electrode_marker = 13
# interface_marker = 15
# wall_poly_marker = 17
# wall_el_marker = 19
# walls_poly = []
# walls_el = []

# for line in gmsh.model.getEntities(dim=1):
#     com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
#     if np.allclose(com, [L/2,L/2,0]):
#         gmsh.model.addPhysicalGroup(line[0],[line[1]],bulk_marker)
#         gmsh.model.setPhysicalName(line[0], bulk_marker, "Bulk electrolyte")
#     elif np.allclose(com, [L/2,0,0]):
#         gmsh.model.addPhysicalGroup(line[0],[line[1]],electrode_marker)
#         gmsh.model.setPhysicalName(line[0], electrode_marker, "Electrode contact")
#     elif np.allclose(com, [L/2,H_CP,0]):
#         gmsh.model.addPhysicalGroup(line[0],[line[1]],interface_marker)
#         gmsh.model.setPhysicalName(line[0], interface_marker, "Interface")
#     elif np.allclose(com, [0,H_CP/2,0]) or np.allclose(com, [L,H_CP/2,0]):
#         walls_poly.append(line[1])
#     elif np.allclose(com, [0,(L-H_CP)/2+H_CP,0]) or np.allclose(com, [L,(L-H_CP)/2+H_CP,0]):
#         walls_el.append(line[1])

# gmsh.model.addPhysicalGroup(1, walls_poly, wall_poly_marker)
# gmsh.model.setPhysicalName(1, wall_poly_marker, "Walls poly")
# gmsh.model.addPhysicalGroup(1,walls_el, wall_el_marker)
# gmsh.model.setPhysicalName(1,wall_el_marker, "Walls electrolyte")

#gmsh.model.occ.synchronize()
# gmsh.model.mesh.generate(dim=2)

domain, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=2)
gmsh.finalize()
with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

#pyvista.start_xvfb()
# plotter = pyvista.Plotter()
# tdim = domain.topology.dim
# domain.topology.create_connectivity(tdim, tdim)
# topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology,cell_types,geometry)
# num_local_cells = domain.topology.index_map(tdim).size_local

# actor = plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     cell_tag_fig = plotter.screenshot("cell_tags.png")
# Create connectivity
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

viskex.dolfinx.plot_mesh(domain)

#domain = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, topology, geometry, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
#cluster = ipp.Cluster(engines="mpi", n=10)
#rc = cluster.start_and_connect_sync()




x = ufl.SpatialCoordinate(domain)

element_family_vector = basix.ElementFamily.BDM
element_degree = 1
variant = basix.LagrangeVariant.equispaced
vector_el = basix.ufl.element(element_family_vector, domain.topology.cell_name(), element_degree, variant)

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
F = assemble_stationary_problem(phi, c1, c2, vphi, v1, v2)

# SET BOUNDARY CONDITIONS
bcs = assemble_boundary_conditions_stationary_electrode_matrix(
    V, facet_tags, fdim, V_bulk_scaled, Vapp_scaled, c_bulk_scaled, bulk_marker, activated_electrodes, counter_electrodes)



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


points_on_proc, cells = extract_central_cutline(domain, L_scaled)


points_gathered = MPI.COMM_WORLD.gather(points_on_proc, root=0)
cells_gathered = MPI.COMM_WORLD.gather(cells, root=0)

c1_local = u.sub(0).eval(points_on_proc, cells)*c_char
c2_local = u.sub(1).eval(points_on_proc, cells)*c_char
phi_local = u.sub(2).eval(points_on_proc, cells)*phi_char

c1_gathered = comm.gather(c1_local,root=0)
c2_gathered = comm.gather(c2_local,root=0)
phi_gathered = comm.gather(phi_local,root=0)



if MPI.COMM_WORLD.rank == 0:  

    points_gathered_filtered = [p for p in points_gathered if p.size > 0]
    if len(points_gathered_filtered) > 0:
        points_combined = np.vstack(points_gathered_filtered)  
    c1_combined = np.vstack(c1_gathered)
    c2_combined = np.vstack(c2_gathered)
    phi_combined = np.vstack(phi_gathered)

    sort_indices = np.argsort(points_combined[:, 2])  
    points_combined = points_combined[sort_indices]
    c1_combined = c1_combined[sort_indices]
    c2_combined = c2_combined[sort_indices]
    phi_combined = phi_combined[sort_indices]

    print(c2_combined[0]/(c_bulk*np.exp(phi_combined[0]/phi_char)))
    fig = plt.figure()
    plt.plot(points_combined[:, 2]*x_char, c1_combined, "k", linewidth=2, label="c1")
    plt.plot(points_combined[:, 2]*x_char, c2_combined, "b", linewidth=2, label="c2")
    
    plt.xscale("linear")
    plt.grid(True)
    plt.xlabel("x")
    plt.legend()
    fig.savefig("Concentrations_SS.png", bbox_inches="tight", dpi=300)

    fig2 = plt.figure()
    plt.plot(points_combined[:, 2]*x_char, phi_combined, "r", linewidth=2, label="phi")
    plt.grid(True)
    plt.xlabel("x")
    plt.xscale("linear")
    plt.legend()
    print("Done")
    fig2.savefig("Potential_SS.png", bbox_inches="tight", dpi=300)

    #plt.show()


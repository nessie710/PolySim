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
assert np.dtype(PETSc.ScalarType).kind == 'c'
from boundary_conditions import assemble_boundary_conditions_stationary, assemble_boundary_conditions_AC
from PNP_equation_sets import assemble_stationary_problem, assemble_AC_problem
from extract_cutlines import extract_central_cutline


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
R_sens = 9e-9
R_sens_scaled = R_sens/x_char
frequencies = np.logspace(3,9,10)/f_char

omegas = 2*np.pi*frequencies



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
    
# Load mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
model = gmsh.model()
bulk_marker, electrode_marker, wall_marker = gmsh_simple_1_electrode_domain(model, "Chamber", L_scaled, R_sens_scaled)
#partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
partitioner = mesh.create_cell_partitioner(partitioner_parmetis())

domain, cell_tags, facet_tags = gmshio.read_from_msh("mesh3D.msh", MPI.COMM_WORLD,0 , gdim=3, partitioner=partitioner)
topology, geometry = domain.topology, domain.geometry

#domain = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, topology, geometry, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
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
bcs = assemble_boundary_conditions_stationary(V, facet_tags, fdim, V_bulk_scaled, Vapp_scaled, c_bulk_scaled, bulk_marker, electrode_marker)



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
    # fig = plt.figure()
    # plt.plot(points_combined[:, 2]*x_char, c1_combined, "k", linewidth=2, label="c1")
    # plt.plot(points_combined[:, 2]*x_char, c2_combined, "b", linewidth=2, label="c2")
    
    # plt.xscale("linear")
    # plt.grid(True)
    # plt.xlabel("x")
    # plt.legend()
    

    # fig2 = plt.figure()
    # plt.plot(points_combined[:, 2]*x_char, phi_combined, "r", linewidth=2, label="phi")
    # plt.grid(True)
    # plt.xlabel("x")
    # plt.xscale("linear")
    # plt.legend()
    # print("Done")
    #plt.show()


# AC study
melAC = basix.ufl.mixed_element([scalar_el, scalar_el, scalar_el])
VAC = fem.functionspace(domain, melAC)
v1ac, v2ac, vphiac= ufl.TestFunctions(VAC)
uac = fem.Function(VAC, dtype=np.complex128)

c1ac, c2ac, phiac = ufl.split(uac)

if MPI.COMM_WORLD.rank == 0:
    capacitances = []
    currents1 = []
    currents2 = []
    c2ac_array = np.zeros((len(frequencies),101))

points_gathered = MPI.COMM_WORLD.gather(points_on_proc, root=0)
cells_gathered = MPI.COMM_WORLD.gather(cells, root=0)

#dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quad},)

for (idx, omega) in enumerate(omegas):
    
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


    G = assemble_AC_problem(phiac, phi, c1ac, c2ac, c1, c2, vphiac, v1ac, v2ac, omega)
    
    bcs = assemble_boundary_conditions_AC(VAC, facet_tags, fdim, V_bulk_scaled, Vapp_AC_scaled, c_bulk_scaled, bulk_marker, electrode_marker, wall_marker)





    problem = NonlinearProblem(G, uac, bcs = bcs)
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


    # SOLVE AC PROBLEM
    solver.solve(uac)



    uac_values = []




    J1 = (-ufl.grad(c1ac) - c1ac*ufl.grad(phi) - c1*ufl.grad(phiac))
    V_vector_temp = fem.functionspace(domain, vector_el)
    J1_func = fem.Function(V_vector_temp)
    expr_J1 = fem.Expression(J1, V_vector_temp.element.interpolation_points())
    J1_func.interpolate(expr_J1)
    form = fem.form(J1_func[2]*ds(electrode_marker))
    integral_j1_value = fem.assemble_scalar(form)
    j1_local = integral_j1_value*J1_char*(x_char**2)

    J2 = (-ufl.grad(c2ac) + c2ac*ufl.grad(phi) + c2*ufl.grad(phiac))
    J2_func = fem.Function(V_vector_temp)
    expr_J2 = fem.Expression(J2, V_vector_temp.element.interpolation_points())
    J2_func.interpolate(expr_J2)
    form = fem.form(J2_func[2]*ds(electrode_marker))
    integral_j2_value = fem.assemble_scalar(form)
    j2_local = integral_j2_value *J2_char*(x_char**2)
    
    
    #if len(uac.sub(2).eval(points_on_proc, cells))>0:
    c1ac_local = uac.sub(0).eval(points_on_proc, cells) * c_char
    c2ac_local = uac.sub(1).eval(points_on_proc, cells) * c_char
    phiac_distributed_local = uac.sub(2).eval(points_on_proc,cells)*phi_char


    field = -ufl.grad(phiac)
    field_func = fem.Function(V_vector_temp)
    expr = fem.Expression(field, V_vector_temp.element.interpolation_points())
    field_func.interpolate(expr)
    form = fem.form(field_func[2]*ds(electrode_marker))
    integral_field_value = fem.assemble_scalar(form)
    
    j_displ_local = 1j*omega*integral_field_value*f_char*epsilon_w*phi_char*x_char*(1/2)

    area = fem.assemble_scalar(fem.form(1 * ds(electrode_marker)))*x_char**2

    j1_gathered = comm.reduce(j1_local, op=MPI.SUM, root=0)
    j2_gathered = comm.reduce(j2_local, op=MPI.SUM, root=0)
    phiac_gathered = comm.gather(phiac_distributed_local, root=0)
    j_displ_gathered = comm.reduce(j_displ_local, op=MPI.SUM, root=0)
    c1ac_gathered = comm.gather(c1ac_local, root=0)
    c2ac_gathered = comm.gather(c2ac_local, root=0)
    phiac_distributed_gathered = comm.gather(phiac_distributed_local, root=0)
    if MPI.COMM_WORLD.rank == 0:  

        points_gathered_filtered = [p for p in points_gathered if p.size > 0]
        if len(points_gathered_filtered) > 0:
            points_combined = np.vstack(points_gathered_filtered) 
        
        #phiac_gathered_filtered = [p for p in phiac_gathered if p.size>0]
        #if len(phiac_gathered_filtered) > 0:
        phiac_combined = np.vstack(phiac_gathered)
        sort_indices = np.argsort(points_combined[:, 2])  
        points_combined = points_combined[sort_indices]
        phiac_combined = phiac_combined[sort_indices]

        jtot1 =  j_displ_gathered + j1_gathered + j2_gathered
        capacitance = np.imag(jtot1/(phiac_combined[0]*(omega*f_char)))
        capacitances.append(capacitance)
        
        
        # Compute concentrations
        c1ac_combined = np.vstack(c1ac_gathered)
        c2ac_combined = np.vstack(c2ac_gathered)

        sort_indices = np.argsort(points_combined[:, 2])  
        points_combined = points_combined[sort_indices]
        c1ac_combined = c1ac_combined[sort_indices]
        c2ac_combined = c2ac_combined[sort_indices]
        
        c2ac_array[idx,:] = np.squeeze(np.real(c2ac_combined))
        # plt.figure()
        # plt.plot(points_combined[:,2], np.squeeze(np.real(c2ac_combined)))
        # plt.show()


if MPI.COMM_WORLD.rank == 0:  

    print("Percentage error on capacitance")
    expected = epsilon_w/x_char
    simulated = np.squeeze(capacitances)[0]/area
    print(str(np.abs(expected-simulated)/expected) + "%")
    
    plt.figure()
    plt.plot(frequencies*f_char, np.squeeze(capacitances))
    plt.xscale("log")
    
    plt.figure()
    for i in c2ac_array:
        plt.plot(points_combined[:,2], i)
        plt.xscale("log")


    plt.show()



    





# %%

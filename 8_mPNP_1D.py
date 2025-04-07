#%% Import statements

from mpi4py import MPI
import ipyparallel as ipp
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem , default_scalar_type, default_real_type, plot
from dolfinx.fem.petsc import assemble_vector, LinearProblem, assemble_matrix, create_matrix, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem import functionspace
import numpy as np
import pandas as pd
import ufl
import basix.ufl
try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)
import matplotlib.pyplot as plt
from lib02_extract_cutlines import extract_central_cutline_1D
from lib06_snes import NonlinearPDE_SNESProblem
# print(PETSc.ScalarType)
# assert np.dtype(PETSc.ScalarType).kind == 'c'
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

def simulate_mPNP(Vapp, Vbulk, concentrations, plot_flag):


    #VARIABLES (EDITABLE)
    D1 = 1e-9
    D2 = 1e-9
    c_bulk_scaled1 = concentrations.c_bulk_scaled1
    c_bulk_scaled2 = concentrations.c_bulk_scaled2
    c_surf_scaled1 = concentrations.c_surf_scaled1
    c_surf_scaled2 = concentrations.c_surf_scaled2
    c_bulk = concentrations.c_bulk
    Vapp = Vapp
    V_bulk = Vbulk
    L = 1e-8
    d = 0.72e-9
    use_surf_bc = concentrations.use_surf_bc
    #save_data = True

    # SCALE VARIABLES
    phi_char = k*T/q
    c_char = 1/(NA*d**3)
    x_char = np.sqrt(epsilon_w*k*T/((q**2)*NA*c_char))  
    J1_char = q*D1*c_char*NA/x_char
    J2_char = q*D2*c_char*NA/x_char
    t_char = x_char**2/D1
    f_char = 1/t_char
    c_bulk_scaled = c_bulk/c_char
    Vapp_scaled = Vapp/phi_char
    V_bulk_scaled = V_bulk/phi_char
    L_scaled = L/x_char



    #%%px
        
    # Define Mesh

    domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(0.0, L_scaled), nx=50000)
    topology, geometry = domain.topology, domain.geometry
    eps = ufl.Constant(domain, np.finfo(float).eps)
    # cluster = ipp.Cluster(engines="mpi", n=1)
    # rc = cluster.start_and_connect_sync()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Create connectivity
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    x = ufl.SpatialCoordinate(domain)

    # Define elements and mixed elements
    element_family_scalar = basix.ElementFamily.P
    element_degree = 2
    variant = basix.LagrangeVariant.equispaced
    scalar_el = basix.ufl.element(element_family_scalar, domain.topology.cell_name(), element_degree, variant)

    mel = basix.ufl.mixed_element([scalar_el, scalar_el, scalar_el])

    V = fem.functionspace(domain, mel)

    v1, v2, vphi= ufl.TestFunctions(V)
    u = fem.Function(V)
    c1, c2, phi= ufl.split(u)


    # Initial conditions
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

    # MODIFIED PNP EQUATION SET

    # F1 = -ufl.inner(ufl.grad(c1)[0], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[0], v1) * ufl.dx - ufl.inner((c1/(1-c1-c2))*(ufl.grad(c1)[0]+ ufl.grad(c2)[0]), v1)*ufl.dx 
    # F2 = -ufl.inner(ufl.grad(c2)[0], v2) * ufl.dx + ufl.inner(c2 * ufl.grad(phi)[0], v2) * ufl.dx - ufl.inner((c2/(1-c1-c2))*(ufl.grad(c1)[0]+ ufl.grad(c2)[0]), v2)*ufl.dx

    F7 = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - ufl.inner((c1-c2), vphi) * ufl.dx
    F8 = ufl.inner(-ufl.grad(c1) - c1 * ufl.grad(phi) - (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v1)) * ufl.dx
    F9 = ufl.inner(-ufl.grad(c2) + c2 * ufl.grad(phi) - (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v2)) * ufl.dx
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


    if use_surf_bc:
        V_split = V.sub(0)
        V_c1, _ = V_split.collapse()
        ud = fem.Function(V_c1)
        ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled1)
        dofs_bulk = fem.locate_dofs_geometrical((V_split,V_c1), boundary_R)
        bc_c1_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)


        V_split = V.sub(1)
        V_c2, _ = V_split.collapse()
        ud = fem.Function(V_c2)
        ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled2)
        dofs_bulk = fem.locate_dofs_geometrical((V_split,V_c2), boundary_R)
        bc_c2_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)  


        V_split = V.sub(0)
        V_c1, _ = V_split.collapse()
        ud = fem.Function(V_c1)
        ud.interpolate(lambda x : x[0]*0 + c_surf_scaled1)
        dofs_bulk = fem.locate_dofs_geometrical((V_split,V_c1), boundary_L)
        bc_c1_surf = fem.dirichletbc(ud, dofs_bulk, V_split)


        V_split = V.sub(1)
        V_c2, _ = V_split.collapse()
        ud = fem.Function(V_c2)
        ud.interpolate(lambda x : x[0]*0 + c_surf_scaled2)
        dofs_bulk = fem.locate_dofs_geometrical((V_split,V_c2), boundary_L)
        bc_c2_surf = fem.dirichletbc(ud, dofs_bulk, V_split)  

        bcs = [bc_potential_bulk, bc_potential_surface, bc_c1_bulk, bc_c2_bulk, bc_c1_surf, bc_c2_surf]
    else:
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

    u.x.scatter_forward()
    dolfinx.log.set_log_level(dolfinx.cpp.log.LogLevel.ERROR)

    # POSTPROCESS

    #points_on_proc, cells = extract_central_cutline_1D(domain, L_scaled)

    points = np.array([[0.5]], dtype=np.float64)
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    cells_local = range(0,num_cells_local-1)

    x_expr = dolfinx.fem.Expression(ufl.SpatialCoordinate(domain), points)
    coords = x_expr.eval(domain, cells_local)
    # coords = coords[0:len(coords)-1]

    points_3D = np.zeros((3,len(coords)))  
    points_3D[0] = coords[:,0]

    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points_3D.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points_3D.T)
    for i, point in enumerate(points_3D.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    points_gathered = MPI.COMM_WORLD.gather(points_on_proc, root=0)
    cells_gathered = MPI.COMM_WORLD.gather(cells, root=0)

    # print(np.shape(points_gathered))
    # print(np.shape(cells_gathered))
    c1_local = u.sub(0).eval(points_gathered[0], cells_gathered[0])*c_char
    c2_local = u.sub(1).eval(points_gathered[0], cells_gathered[0])*c_char
    phi_local = u.sub(2).eval(points_gathered[0], cells_gathered[0])*phi_char

    J1 = -ufl.grad(c1) - c1 * ufl.grad(phi) - (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))
    expr_J1 = fem.Expression(J1, points)
    j1_local = expr_J1.eval(domain,cells_local)*J1_char*(x_char**2)

    J2 = -ufl.grad(c2) + c2 * ufl.grad(phi) - (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2))
    expr_J2 = fem.Expression(J2,points)
    j2_local = expr_J2.eval(domain,cells_local)*J2_char*(x_char**2)




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

        sort_indices = np.argsort(points_combined[:, 0])
        points_combined = points_combined[sort_indices]
        c1_combined = c1_combined[sort_indices]
        c2_combined = c2_combined[sort_indices]
        phi_combined = phi_combined[sort_indices]
        
        # print("Simulated concentration")
        # print(c2_combined[0])
        # print("Theoretical concentration")
        # print(c_bulk*np.exp(phi_combined[0]/(k*T/q)) / (1- 2*NA*d**3*c_bulk + 2*NA*d**3*c_bulk*np.cosh(phi_combined[0]/(k*T/q))))
        # print("PNP concentration")
        # print(c_bulk*np.exp(phi_combined[0]/(k*T/q)))

        if plot_flag:
            fig = plt.figure()
            plt.plot(points_combined[:, 0]*x_char, c1_combined, "k", linewidth=2, label="c1")
            plt.plot(points_combined[:, 0]*x_char, c2_combined, "b", linewidth=2, label="c2")
            #plt.plot(points_combined[:,0]*x_char, c_bulk*np.exp(phi_combined/(k*T/q)) / (1- 2*NA*d**3*c_bulk + 2*NA*d**3*c_bulk*np.cosh(phi_combined/(k*T/q))), "green", label="c2 theoretical")
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


            fig3 = plt.figure()
            plt.plot(coords*x_char, j1_combined+j2_combined, "black", linewidth=2, label="jtot")
            plt.plot(coords*x_char, j1_combined, "r", linewidth=2, label="j1")
            plt.plot(coords*x_char, j2_combined, "b", linewidth=2, label="j2")
            plt.xlabel("x")
            plt.grid(True)
            plt.xscale("linear")
            plt.yscale("linear")
            plt.legend()

            fig4 = plt.figure()
            plt.plot(points_combined[:, 0]*x_char, np.gradient(phi_combined[:,0]), "r", linewidth=2, label="field")
            plt.grid(True)
            plt.xlabel("x")
            plt.xscale("linear")
            plt.legend()

            print("Done")
            plt.show()

        column_names = ["x", "Potential", "J1", "J2", "C1", "C2"]
        data = np.hstack((coords, phi_combined, j1_combined, j2_combined, c1_combined, c2_combined))

        df = pd.DataFrame(data=data,columns=column_names)

        return df







if __name__ == "__main__":
    class Concentration_BC():
        def __init__(self,c_bulk, c_bulk1, c_bulk2, c_surf1, c_surf2, use_surf_bc):
            self.c_bulk = c_bulk
            self.c_bulk_scaled1 = c_bulk1
            self.c_bulk_scaled2 = c_bulk2
            self.c_surf_scaled1 = c_surf1
            self.c_surf_scaled2 = c_surf2
            self.use_surf_bc = use_surf_bc

    # Simulation conditions
    Vapp = k*T/q
    concentrations = Concentration_BC(170, c_bulk1=0.4, c_bulk2=0.4, c_surf1=0.8, c_surf2=0.1, use_surf_bc=True)
    simulate_mPNP(Vapp, 0,concentrations, True)
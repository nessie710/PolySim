#%% Import statements

from mpi4py import MPI
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

from simple_mesh_elements import gmsh_square
from create_mesh_function import create_mesh
import matplotlib.pyplot as plt

import dolfinx.fem.petsc
import basix.ufl
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver

print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'



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
c_char = 1
x_char = np.sqrt(epsilon_w*k*T/((q**2)*2*c_bulk*NA))
J1_char = q*D1*c_char/x_char
J2_char = q*D2*c_char/x_char
t_char = x_char**2/np.sqrt(D1*D2)
f_char = 1/t_char
omega_char = 2*np.pi*f_char


Vapp = 1e-3
Vapp_scaled = Vapp/phi_char
V_bulk = 0
V_bulk_scaled = V_bulk/phi_char
Vapp_AC = 1e-3
Vapp_AC_scaled = Vapp_AC/phi_char
L = 5e-8
L_scaled = L/x_char


#%% Generate mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
model = gmsh.model()

model = gmsh_square(model, "Square", L_scaled)
model.setCurrent("Square")
create_mesh(MPI.COMM_WORLD, model, "Square", f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf","w")

with XDMFFile(MPI.COMM_WORLD, "out_gmsh/mesh_rank_0.xdmf","r") as file:
    domain = file.read_mesh(name="Square")

topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
#plotter.show()

#domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

#%% Define model

# Create connectivity
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
x = ufl.SpatialCoordinate(domain)
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))

element_family_vector = basix.ElementFamily.BDM
element_degree = 1
variant = basix.LagrangeVariant.equispaced
vector_el = basix.ufl.element(element_family_vector, domain.topology.cell_name(), element_degree, variant)

element_family_scalar = basix.ElementFamily.P
element_degree = 1
variant = basix.LagrangeVariant.equispaced
scalar_el = basix.ufl.element(element_family_scalar, domain.topology.cell_name(), element_degree, variant)

mel = basix.ufl.mixed_element([scalar_el, scalar_el, scalar_el])

V = fem.functionspace(domain, mel)

v1, v2, vphi= ufl.TestFunctions(V)
u = fem.Function(V)

c1, c2, phi= ufl.split(u)



n = ufl.FacetNormal(domain)


# # Variational formulation scaled

F1 = -ufl.inner(ufl.grad(c1)[0], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[0], v1) * ufl.dx
F2 = -ufl.inner(ufl.grad(c1)[1], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[1], v1) * ufl.dx
F3 = ufl.inner(ufl.grad(c2)[0], v2) * ufl.dx - ufl.inner(c2 * ufl.grad(phi)[0], v2) * ufl.dx
F4 = ufl.inner(ufl.grad(c2)[1], v2) * ufl.dx - ufl.inner(c2 * ufl.grad(phi)[1], v2) * ufl.dx
F5 = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - ufl.inner((c1-c2), vphi) * ufl.dx
F6 = ufl.inner(-ufl.grad(c1) - c1 * ufl.grad(phi), ufl.grad(v1)) * ufl.dx
F7 = ufl.inner(ufl.grad(c2) - c2 * ufl.grad(phi), ufl.grad(v2)) * ufl.dx

F = F1 + F2 + F3 + F4 + F5 + F6 + F7 

# %% SET BOUNDARY CONDITIONS

def allsides(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], L_scaled)), np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], L_scaled)))

def bulk(x):
    return np.isclose(x[0],0)

def surface(x):
    return np.isclose(x[0],L_scaled)


def sides(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], L_scaled))

def allsides_except_bulk(x):
    return np.logical_or(np.isclose(x[0], L_scaled), np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], L_scaled)))

def allsides_except_surface(x):
    return np.logical_or(np.isclose(x[0], L_scaled), np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], L_scaled)))


boundaries = [(1, bulk),
              (2, surface),
              (3, sides),
              (4, allsides_except_bulk),
              (5, allsides)]


facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


class PotentialBoundaryCondition():
    def __init__(self, type, values, marker):
        self._type = type
        if type == "Dirichlet":

            # Get subspace
            V_split = V.sub(2)
            facets = facet_tag.find(marker)
            V1, _ = V_split.collapse()
            dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
            u_D = fem.Function(V1)
            u_D.interpolate(values)
            self._bc = fem.dirichletbc(u_D, dofs, V_split)

        elif type == "Neumann":
                V_split = V.sub(2)
                facets = facet_tag.find(marker)
                V1, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                u_N = fem.Function(V1)
                u_N.interpolate(values)
                self._bc = ufl.inner(u_N, vphi) * ds(marker) 
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type


class NaBoundaryCondition():
    def __init__(self, type, values, marker):
        self._type = type
        if type == "Dirichlet":

            # Get subspace
            V_split = V.sub(0)
            facets = facet_tag.find(marker)
            V2, _ = V_split.collapse()
            dofs = fem.locate_dofs_topological((V_split, V2), fdim, facets)
            u_D = fem.Function(V2)
            u_D.interpolate(values)
            self._bc = fem.dirichletbc(u_D, dofs, V_split)

        elif type == "Neumann":
            V_split = V.sub(0)
            facets = facet_tag.find(marker)
            V2, _ = V_split.collapse()
            dofs = fem.locate_dofs_topological((V_split, V2), fdim, facets)
            u_N = fem.Function(V2)
            u_N.interpolate(values)
            self._bc = ufl.inner(u_N, v1) * ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type
    

class ClBoundaryCondition():
    def __init__(self, type, values, marker):
        self._type = type
        if type == "Dirichlet":

            # Get subspace
            V_split = V.sub(1)
            facets = facet_tag.find(marker)
            V3, _ = V_split.collapse()
            dofs = fem.locate_dofs_topological((V_split, V3), fdim, facets)
            u_D = fem.Function(V3)
            u_D.interpolate(values)
            self._bc = fem.dirichletbc(u_D, dofs, V_split)

        elif type == "Neumann":
                V_split = V.sub(1)
                facets = facet_tag.find(marker)
                V3, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V3), fdim, facets)
                u_N = fem.Function(V3)
                u_N.interpolate(values)
                self._bc = ufl.inner(u_N, v2) * ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

    
boundary_conditions = [
    PotentialBoundaryCondition("Dirichlet", lambda x: x[0]*0 + V_bulk_scaled, 1),
    PotentialBoundaryCondition("Dirichlet", lambda x: x[0]*0 + Vapp_scaled, 2),
    PotentialBoundaryCondition("Neumann", lambda x: x[0]*0, 3),
    NaBoundaryCondition("Dirichlet", lambda x: x[0]*0 + c_bulk, 1),
    NaBoundaryCondition("Neumann", lambda x: x[0]*0, 4),
    ClBoundaryCondition("Dirichlet", lambda x: x[0]*0 + c_bulk, 1),
    ClBoundaryCondition("Neumann", lambda x: x[0]*0, 4)
]

bcs = []

for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        if isinstance(condition,ClBoundaryCondition):
            F -= condition.bc
        else:
            F += condition.bc




problem = NonlinearProblem(F, u, bcs = bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
solver.max_it = 200


# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()  # type: ignore
# For factorisation prefer superlu_dist, then MUMPS, then default
# if sys.hasExternalPackage("superlu_dist"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
# elif sys.hasExternalPackage("mumps"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

dolfinx.log.set_log_level(dolfinx.cpp.log.LogLevel.INFO)

solver.solve(u)

# EXTRACT CENTRAL CUTLINE
tol = 0.001
x = np.linspace(0, L_scaled-tol, 101)
points = np.zeros((3,101))
points[0] = x
points[1] = np.ones((1,101))*0.5
u_values = []

bb_tree = gm.bb_tree(domain, domain.topology.dim)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = gm.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = gm.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
c1_values = u.sub(0).eval(points_on_proc, cells)*c_char
c2_values = u.sub(1).eval(points_on_proc, cells)*c_char
phi_values = u.sub(2).eval(points_on_proc, cells)*phi_char


fig = plt.figure()
plt.plot(points_on_proc[:, 0]*x_char, c1_values, "k", linewidth=2, label="c1")
plt.plot(points_on_proc[:, 0]*x_char, c2_values, "b", linewidth=2, label="c2")
plt.yscale("linear")
plt.grid(True)
plt.xlabel("x")
plt.legend()


fig2 = plt.figure()
plt.plot(points_on_proc[:, 0]*x_char, phi_values, "r", linewidth=2, label="phi")
plt.grid(True)
plt.xlabel("x")
plt.legend()
plt.show()

#%% AC study
melAC = basix.ufl.mixed_element([scalar_el, scalar_el, scalar_el, vector_el, vector_el])
VAC = fem.functionspace(domain, melAC)

v1ac, v2ac, vphiac, vj1AC, vj2AC= ufl.TestFunctions(VAC)
uac = fem.Function(VAC, dtype=np.complex128)

c1ac, c2ac, phiac, J1ac, J2ac = ufl.split(uac)

frequencies = np.logspace(3,9,10)/f_char
omegas = 2*np.pi*frequencies
capacitances = []
conductances = []
# Variational formulation

tol = 0.001
points = np.zeros((3,1))
points[0] = L_scaled
points[1] = L_scaled/2
points[2] = 0
bb_tree = gm.bb_tree(domain, domain.topology.dim)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = gm.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = gm.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)


for omega in omegas:

    G1 = ufl.inner(c1ac*omega*1j, v1ac) * ufl.dx - ufl.inner(J1ac, ufl.grad(v1ac))*ufl.dx
    G2 = ufl.inner(c2ac*omega*1j, v2ac) * ufl.dx + ufl.inner(J2ac, ufl.grad(v2ac))*ufl.dx
    G3 = ufl.inner(ufl.grad(phiac), ufl.grad(vphiac))*ufl.dx - ufl.inner((c1ac - c2ac), vphiac)*ufl.dx
    G4 = ufl.inner(J1ac, vj1AC) * ufl.dx - ufl.inner(-ufl.grad(c1ac) - c1ac*ufl.grad(phi) - c1*ufl.grad(phiac), vj1AC) * ufl.dx
    G5 = ufl.inner(J2ac, vj2AC) * ufl.dx - ufl.inner(ufl.grad(c2ac) - c2ac*ufl.grad(phi) - c2*ufl.grad(phiac), vj2AC) * ufl.dx

    G = G1 + G2 + G3 + G4 + G5

    boundaries = [(1, bulk),
                (2, surface),
                (3, sides),
                (4, allsides_except_bulk),
                (5, allsides_except_surface)]


    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


    class PotentialACBoundaryCondition():
        def __init__(self, type, values, marker):
            self._type = type
            if type == "Dirichlet":

                # Get subspace
                V_split = VAC.sub(2)
                facets = facet_tag.find(marker)
                V1, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                u_D = fem.Function(V1)
                u_D.interpolate(values)
                self._bc = fem.dirichletbc(u_D, dofs, V_split)

            elif type == "Neumann":
                    V_split = VAC.sub(2)
                    facets = facet_tag.find(marker)
                    V1, _ = V_split.collapse()
                    dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                    u_N = fem.Function(V1)
                    u_N.interpolate(values)
                    self._bc = ufl.inner(u_N, vphiac) * ds(marker) 
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type

    class NaACBoundaryCondition():
        def __init__(self, type, values, marker):
            self._type = type
            if type == "Dirichlet":

                # Get subspace
                V_split = VAC.sub(0)
                facets = facet_tag.find(marker)
                V2, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V2), fdim, facets)
                u_D = fem.Function(V2)
                u_D.interpolate(values)
                self._bc = fem.dirichletbc(u_D, dofs, V_split)

            elif type == "Neumann":
                V_split = VAC.sub(0)
                facets = facet_tag.find(marker)
                V2, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V2), fdim, facets)
                u_N = fem.Function(V2)
                u_N.interpolate(values)
                self._bc = ufl.inner(u_N, v1ac) * ds(marker)
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type
        
    class ClACBoundaryCondition():
        def __init__(self, type, values, marker):
            self._type = type
            if type == "Dirichlet":

                # Get subspace
                V_split = VAC.sub(1)
                facets = facet_tag.find(marker)
                V3, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V3), fdim, facets)
                u_D = fem.Function(V3)
                u_D.interpolate(values)
                self._bc = fem.dirichletbc(u_D, dofs, V_split)

            elif type == "Neumann":
                    V_split = V.sub(1)
                    facets = facet_tag.find(marker)
                    V3, _ = V_split.collapse()
                    dofs = fem.locate_dofs_topological((V_split, V3), fdim, facets)
                    u_N = fem.Function(V3)
                    u_N.interpolate(values)
                    self._bc = ufl.inner(u_N, v2ac) * ds(marker)
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type
   
    class J1ACBoundaryCondition():
        def __init__(self, type, values, marker):
            self._type = type
            if type == "Dirichlet":

                # Get subspace
                V_split = VAC.sub(3)
                facets = facet_tag.find(marker)
                V1, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                u_D = fem.Function(V1)
                u_D.interpolate(values)
                self._bc = fem.dirichletbc(u_D, dofs, V_split)

            elif type == "Neumann":
                    V_split = VAC.sub(3)
                    facets = facet_tag.find(marker)
                    V1, _ = V_split.collapse()
                    dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                    u_N = fem.Function(V1)
                    u_N.interpolate(values)
                    self._bc = ufl.inner(u_N, vj1AC) * ds(marker) 
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type
        
    class J2ACBoundaryCondition():
        def __init__(self, type, values, marker):
            self._type = type
            if type == "Dirichlet":

                # Get subspace
                V_split = VAC.sub(4)
                facets = facet_tag.find(marker)
                V1, _ = V_split.collapse()
                dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                u_D = fem.Function(V1)
                u_D.interpolate(values)
                self._bc = fem.dirichletbc(u_D, dofs, V_split)

            elif type == "Neumann":
                    V_split = VAC.sub(4)
                    facets = facet_tag.find(marker)
                    V1, _ = V_split.collapse()
                    dofs = fem.locate_dofs_topological((V_split, V1), fdim, facets)
                    u_N = fem.Function(V1)
                    u_N.interpolate(values)
                    self._bc = ufl.inner(u_N, vj2AC) * ds(marker) 
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
        @property
        def bc(self):
            return self._bc

        @property
        def type(self):
            return self._type
        

    boundary_conditions = [
        PotentialACBoundaryCondition("Dirichlet", lambda x: x[0]*0, 1),
        PotentialACBoundaryCondition("Dirichlet", lambda x: x[0]*0 + Vapp_AC_scaled, 2),
        PotentialACBoundaryCondition("Neumann", lambda x: x[0]*0, 3),
        NaACBoundaryCondition("Dirichlet", lambda x: x[0]*0, 1),
        NaACBoundaryCondition("Neumann", lambda x: x[0]*0, 4),
        ClACBoundaryCondition("Dirichlet", lambda x: x[0]*0, 1),
        ClACBoundaryCondition("Neumann", lambda x: x[0]*0, 4),
        J1ACBoundaryCondition("Neumann", lambda x: x[0]*np.zeros((3,1)), 3),
        J2ACBoundaryCondition("Neumann", lambda x: x[0]*np.zeros((3,1)), 3),
    ]

    bcs = []

    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)
        else:
            if isinstance(condition,ClACBoundaryCondition):
                G -= condition.bc
            else:
                G += condition.bc


    problem = NonlinearProblem(G, uac, bcs = bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
    solver.max_it = 200


    # We can customize the linear solver used inside the NewtonSolver by
    # modifying the PETSc options
    ksp = solver.krylov_solver
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    sys = PETSc.Sys()  # type: ignore
    # For factorisation prefer superlu_dist, then MUMPS, then default
    # if sys.hasExternalPackage("superlu_dist"):
    #     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
    # elif sys.hasExternalPackage("mumps"):
    #     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    dolfinx.log.set_log_level(dolfinx.cpp.log.LogLevel.INFO)

    solver.solve(uac)



    uac_values = []


    
    
    x_grad = ufl.inner(ufl.as_vector((x[2], x[1],x[0])), ufl.grad(uac.sub(2)))
    W = fem.functionspace(domain, ("DQ", 1))
    expr = fem.Expression(x_grad, W.element.interpolation_points())
    w = fem.Function(W)
    w.interpolate(expr)

    j1_values = uac.sub(3).eval(points_on_proc, cells)[0] * J1_char
    j2_values = uac.sub(4).eval(points_on_proc, cells)[0] * J2_char
    phiac_values = uac.sub(2).eval(points_on_proc, cells) * phi_char
    phi0_values = u.sub(2).eval(points_on_proc, cells) * phi_char
    #jdispl_values = -1j*omega*epsilon_w*w


    jtot =  j1_values + j2_values
    
    capacitance = np.imag(-jtot/(phiac_values*omega))
    conductance = np.real(-jtot/phiac_values)
    
    conductances.append(conductance)
    capacitances.append(capacitance)

plt.figure()
plt.plot(frequencies*f_char, capacitances)
plt.xscale("log")
plt.show()

# plt.figure()
# plt.plot(frequencies*f_char, conductances)
# plt.xscale("log")
# plt.show()

#%% POSTPROCESS

# EXTRACT CENTRAL CUTLINE
# tol = 0.001
# x = np.linspace(0, L-tol, 101)
# points = np.zeros((3,101))
# points[0] = x
# points[1] = np.ones((1,101))*0.5
# u_values = []

# bb_tree = gm.bb_tree(domain, domain.topology.dim)
# cells = []
# points_on_proc = []
# # Find cells whose bounding-box collide with the the points
# cell_candidates = gm.compute_collisions_points(bb_tree, points.T)
# # Choose one of the cells that contains the point
# colliding_cells = gm.compute_colliding_cells(domain, cell_candidates, points.T)
# for i, point in enumerate(points.T):
#     if len(colliding_cells.links(i)) > 0:
#         points_on_proc.append(point)
#         cells.append(colliding_cells.links(i)[0])

# points_on_proc = np.array(points_on_proc, dtype=np.float64)
# c1_values = u.sub(0).eval(points_on_proc, cells)
# c2_values = u.sub(1).eval(points_on_proc, cells)
# phi_values = u.sub(2).eval(points_on_proc, cells)


# fig = plt.figure()
# plt.plot(points_on_proc[:, 0], c1_values, "k", linewidth=2, label="c1")
# plt.plot(points_on_proc[:, 0], c2_values, "b", linewidth=2, label="c2")
# plt.yscale("linear")
# plt.grid(True)
# plt.xlabel("x")
# plt.legend()


# fig2 = plt.figure()
# plt.plot(points_on_proc[:, 0], phi_values, "r", linewidth=2, label="phi")
# plt.grid(True)
# plt.xlabel("x")
# plt.legend()
# plt.show()


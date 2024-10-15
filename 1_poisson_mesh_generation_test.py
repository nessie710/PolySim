#%% Import statements

from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
from dolfinx import mesh, fem
from dolfinx.fem import functionspace
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx import geometry as gm
from dolfinx.fem.petsc import LinearProblem
import pyvista
from dolfinx import plot
try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)

from simple_mesh_elements import gmsh_square
from create_mesh_function import create_mesh
import matplotlib.pyplot as plt


#%% Generate mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
model = gmsh.model()

L = 1

model = gmsh_square(model, "Square", L)
model.setCurrent("Square")
create_mesh(MPI.COMM_WORLD, model, "Square", f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf","w")

with XDMFFile(MPI.COMM_WORLD, "out_gmsh/mesh_rank_0.xdmf","r") as file:
    domain = file.read_mesh(name="Square")

topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()


#%% Define model
# Define the model's function space 
V = functionspace(domain, ("Lagrange",1))

# Create connectivity
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

class ChargeDelta:
    def __init__(self,eps):
        self.eps = eps
    
    def eval(self, x):
        return (1/(self.eps*np.sqrt(np.pi)))*np.exp(-((x[0]-1)/self.eps)**2)
    
class ChargeStep:
    def __init__(self, tol):
        self.tol = tol

    def eval(self, x):
        return 1*(x[0]>(1-self.tol))
        
f = fem.Function(V)        
rho = ChargeStep(1e-8)
f.interpolate(rho.eval)

# Define normal vector
n = ufl.FacetNormal(domain)
# Define Neumann boundary condition
g = -ufl.dot(n, ufl.grad(f))
# Define physical constants and problem expression
epsr = 80
eps0 = 8.85e-12
epsilon = fem.Constant(domain, default_scalar_type(1))
F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(f, v) * ufl.dx

# Sort-out boundary names and locations
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)))]

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

class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(V)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            self._bc = fem.dirichletbc(u_D, dofs)
        elif type == "Neumann":
                self._bc = ufl.inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * ufl.inner(u-values[1], v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

# Define the Dirichlet condition
dirichlet0 = lambda x: 0*x[0]
dirichlet1 = lambda x: 1+0*x[0]
# Define boundary conditions
boundary_conditions = [BoundaryCondition("Dirichlet", 1, dirichlet0),
                       BoundaryCondition("Dirichlet", 2, dirichlet1),
                       BoundaryCondition("Neumann", 3, g)]

bcs = []
for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        F += condition.bc

# Assemble problem and solve
a = ufl.lhs(F)
L = ufl.rhs(F)
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
print("Solved !")

# Plot solution
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()

# EXTRACT CENTRAL CUTLINE
tol = 0.001
x = np.linspace(0, 1-tol, 101)
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
u_values = uh.eval(points_on_proc, cells)


fig = plt.figure()
plt.plot(points_on_proc[:, 0], u_values, "k", linewidth=2, label="Potential")
plt.plot(points_on_proc[:, 0], f.eval(points_on_proc, cells), "b", linewidth=2, label="Charge")
plt.grid(True)
plt.xlabel("x")
plt.legend()
plt.show()
# If run in parallel as a python file, we save a plot per processor
#plt.savefig(f"membrane_rank{MPI.COMM_WORLD.rank:d}.png")

# EXTRACT VERTICAL CUTLINE
tol = 0.001
y = np.linspace(0, 1-tol, 101)
points = np.zeros((3,101))
points[0] = np.ones((1,101))
points[1] = y
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
u_values = uh.eval(points_on_proc, cells)


fig = plt.figure()
plt.plot(points_on_proc[:, 1], u_values, "k", linewidth=2, label="Potential (V)")
plt.grid(True)
plt.xlabel("y")
plt.legend()
plt.show()
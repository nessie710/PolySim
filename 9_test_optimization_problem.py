




#%% Import statements
import scipy
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
from extract_cutlines import extract_central_cutline_1D
class NonLinearSolver(scipy.sparse.linalg.LinearOperator):
    def __init__(self, F, uh, bcs):
        """
        Solve the problem F(uh, v)=0 forall v
        """
        jacobian = ufl.derivative(F, uh)
        self.J_compiled = dolfinx.fem.form(jacobian)
        self.F_compiled = dolfinx.fem.form(F)
        self.bcs = bcs
        self.A = dolfinx.fem.create_matrix(self.J_compiled)
        self.b = dolfinx.fem.Function(uh.function_space)
        self._A_scipy = self.A.to_scipy()
        self.uh = uh

        # Scipy specific parameters
        self.shape = (len(uh.x.array), len(uh.x.array))
        self.dtype = uh.x.array.dtype
        self.update(uh.x.array, None)

    def update(self, x, f):
        """Update and invert Jacobian"""
        self.A.data[:] = 0
        self.uh.x.array[:] = x
        dolfinx.fem.assemble_matrix(self.A, self.J_compiled, bcs=self.bcs)
        self._A_inv = scipy.sparse.linalg.splu(self._A_scipy)

    def _matvec(self, x):
        """Compute J^{-1}x"""
        return self._A_inv.solve(x)

    def _compute_residual(self, x):
        """
        Evaluate the residual F(x) = 0
        Args:
            x: Input vector with current solution
        Returns:
            Residual array
        """
        self.uh.x.array[:] = x
        self.b.x.array[:] = 0
        dolfinx.fem.assemble_vector(self.b.x.array, self.F_compiled)
        dolfinx.fem.apply_lifting(self.b.x.array, [self.J_compiled], [self.bcs], x0=[self.uh.x.array], alpha=-1.0)
        self.b.x.scatter_reverse(dolfinx.la.InsertMode.add)
        [bc.set(self.b.x.array, x0=self.uh.x.array, alpha=-1.0) for bc in self.bcs]
        return self.b.x.array

    def linSolver(self, _A, x, **kwargs):
        """
        The linear solver method we will use.
        Simply return `J^-1 x` for an input x
        """
        return kwargs["M"]._matvec(x), 0

    def solve(self, maxiter: int = 100, verbose: bool = False):
        """Call Newton-Krylov solver with direct solving (pre-conditioning only)"""
        self.uh.x.array[:] = scipy.optimize.newton_krylov(
            self._compute_residual,
            self.uh.x.array,
            method=self.linSolver,
            verbose=verbose,
            line_search=None,
            maxiter=maxiter,
            inner_M=self,
        )


class LinearSolver:
    def __init__(self, a, L, uh, bcs):
        self.a_compiled = dolfinx.fem.form(a)
        self.L_compiled = dolfinx.fem.form(L)
        self.A = dolfinx.fem.create_matrix(self.a_compiled)
        self.b = dolfinx.fem.Function(uh.function_space)
        self.bcs = bcs
        self._A_scipy = self.A.to_scipy()
        self.uh = uh

    def solve(self):
        self._A_scipy.data[:] = 0

        dolfinx.fem.assemble_matrix(self.A, self.a_compiled, bcs=self.bcs)

        self.b.x.array[:] = 0
        dolfinx.fem.assemble_vector(self.b.x.array, self.L_compiled)
        dolfinx.fem.apply_lifting(self.b.x.array, [self.a_compiled], [self.bcs])
        self.b.x.scatter_reverse(dolfinx.la.InsertMode.add)
        [bc.set(self.b.x.array) for bc in self.bcs]

        A_inv = scipy.sparse.linalg.splu(self._A_scipy)
        self.uh.x.array[:] = A_inv.solve(self.b.x.array)
        return self.uh



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

d = 0.25e-9

phi_char = k*T/q
c_char = 1/(NA*d**3)
x_char = np.sqrt(epsilon_w*k*T/((q**2)*NA*c_char))  
J1_char = q*D1*c_char*NA/x_char
J2_char = q*D2*c_char*NA/x_char
t_char = x_char**2/D1
f_char = 1/t_char

c_bulk = 170
c_bulk_scaled = c_bulk/c_char
Vapp = 0.5
Vapp_scaled = Vapp/phi_char
V_bulk = 0
V_bulk_scaled = V_bulk/phi_char
L = 1e-7
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

domain = mesh.create_interval(comm=MPI.COMM_WORLD, points=(0.0, L_scaled), nx=5000)
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
s = fem.Function(V)
s1,s2,sphi = ufl.split(s)

n = ufl.FacetNormal(domain)

# IMPORT STATIONARY EQUATION SET

F1 = -ufl.inner(ufl.grad(c1)[0], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[0], v1) * ufl.dx - ufl.inner((c1/(1-c1-c2))*(ufl.grad(c1)[0]+ ufl.grad(c2)[0]), v1)*ufl.dx 
F2 = -ufl.inner(ufl.grad(c2)[0], v2) * ufl.dx + ufl.inner(c2 * ufl.grad(phi)[0], v2) * ufl.dx - ufl.inner((c2/(1-c1-c2))*(ufl.grad(c1)[0]+ ufl.grad(c2)[0]), v2)*ufl.dx

F7 = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - ufl.inner((c1-c2), vphi) * ufl.dx
F8 = ufl.inner(-ufl.grad(c1) - c1 * ufl.grad(phi) - (c1/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v1)) * ufl.dx
F9 = ufl.inner(-ufl.grad(c2) + c2 * ufl.grad(phi) - (c2/(1-c1-c2))*(ufl.grad(c1)+ ufl.grad(c2)), ufl.grad(v2)) * ufl.dx
F8_new = -ufl.inner(ufl.grad(c1),ufl.grad(v1))*ufl.dx + ufl.inner(ufl.dot(ufl.grad(c1), ufl.grad(phi) + (ufl.grad(c1) + ufl.grad(c2))/(1-c1-c2)),v1)*ufl.dx - ufl.inner((c1*ufl.grad(phi)),ufl.grad(v1))*ufl.dx + ufl.inner(c1*ufl.dot(ufl.grad(1/(1-c1-c2)), (ufl.grad(c1)+ufl.grad(c2))),v1)*ufl.dx + ufl.inner(((1/(1-c1-c2))*(ufl.grad(c1)+ufl.grad(c2))),ufl.grad(v1))*ufl.dx
F9_new = -ufl.inner(ufl.grad(c2),ufl.grad(v2))*ufl.dx + ufl.inner(ufl.dot(ufl.grad(c2), -ufl.grad(phi) + (ufl.grad(c1) + ufl.grad(c2))/(1-c1-c2)),v2)*ufl.dx + ufl.inner((c2*ufl.grad(phi)),ufl.grad(v2))*ufl.dx + ufl.inner(c2*ufl.dot(ufl.grad(1/(1-c1-c2)), (ufl.grad(c1)+ufl.grad(c2))),v2)*ufl.dx + ufl.inner(((1/(1-c1-c2))*(ufl.grad(c1)+ufl.grad(c2))),ufl.grad(v2))*ufl.dx


alpha = dolfinx.fem.Constant(domain, 1e-3)
x = ufl.SpatialCoordinate(domain)
d = 1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
J1 = 1 / 2 * (c1 - s1) * (c1 - s1) * ufl.dx 
J1 = 1 / 2 * (c2 - s2) * (c2 - s2) * ufl.dx 
tdim = domain.topology.dim

domain.topology.create_connectivity(tdim - 1, tdim)
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


forward_problem = NonLinearSolver(F, uh=u, bcs=[bc])
lmbda = dolfinx.fem.Function(V)
dFdu = ufl.derivative(F, u, du)
dFdu_adj = ufl.adjoint(dFdu)
dJdu = ufl.derivative(J, u, v)

adj_problem = LinearSolver(ufl.replace(dFdu_adj, {uh: v}), -dJdu, lmbda, [bc])
q = ufl.TrialFunction(W)
dJdf = ufl.derivative(J, f, q)
dFdf = ufl.action(ufl.adjoint(ufl.derivative(F, f, q)), lmbda)
dJdf_compiled = dolfinx.fem.form(dJdf)
dFdf_compiled = dolfinx.fem.form(dFdf)
dLdf = dolfinx.fem.Function(W)
Jh = dolfinx.fem.form(J)


def eval_J(x):
    f.x.array[:] = x
    forward_problem.solve(verbose=False)
    local_J = dolfinx.fem.assemble_scalar(Jh)
    return domain.comm.allreduce(local_J, op=MPI.SUM)

def eval_gradient(x):
    f.x.array[:] = x
    forward_problem.solve()
    adj_problem.solve()
    dLdf.x.array[:] = 0
    dolfinx.fem.assemble_vector(dLdf.x.array, dJdf_compiled)
    dolfinx.fem.assemble_vector(dLdf.x.array, dFdf_compiled)
    return dLdf.x.array


def callback(intermediate_result):
    fval = intermediate_result.fun
    print(f"J: {fval}")


from scipy.optimize import minimize

opt_sol = minimize(
    eval_J,
    f.x.array,
    jac=eval_gradient,
    method="CG",
    tol=1e-9,
    options={"disp": True},
    callback=callback,
)

f.x.array[:] = opt_sol.x
forward_problem.solve()




def f_exact(mod, x):
    return 1 / (1 + 4 * float(alpha) * mod.pi**4) * mod.sin(mod.pi * x[0]) * mod.sin(mod.pi * x[1])

def u_exact(mod, x):
    return 1 / (2 * np.pi**2) * f_exact(mod, x)

u_ex = dolfinx.fem.Function(V)
u_ex.interpolate(lambda x: u_exact(np, x))

L2_error = dolfinx.fem.form(ufl.inner(uh - u_exact(ufl, x), uh - u_exact(ufl, x)) * ufl.dx)
local_error = dolfinx.fem.assemble_scalar(L2_error)
global_error = np.sqrt(domain.comm.allreduce(local_error, op=MPI.SUM))
print(f"Error: {global_error:.2f}")
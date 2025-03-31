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
L = 10e-6
L_scaled = L/x_char
R_sens = 90e-9
R_sens_scaled = R_sens/x_char
R_cylinder = 1e-8
R_cylinder_scaled = R_cylinder/x_char
L_cylinder = 5e-8
L_cylinder_scaled = L_cylinder/x_char
dz = 3e-8
dz_scaled = dz/x_char
pitch = 600e-9
pitch_scaled = pitch/x_char

frequencies = np.logspace(3,9,10)/f_char

omegas = 2*np.pi*frequencies



gmsh.initialize()
gmsh.option.setNumber("General.Terminal",0)
model = gmsh.model()
bulk_marker, electrode_marker, wall_marker = gmsh_simple_1_electrode_domain(model, "Chamber", L_scaled, R_sens_scaled)

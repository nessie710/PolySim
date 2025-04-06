simulator = __import__("8_mPNP_1D")
import numpy as np
import scipy


class Concentration_BC():
    def __init__(self,c_bulk, c_bulk1, c_bulk2, c_surf1, c_surf2, use_surf_bc):
        self.c_bulk = c_bulk
        self.c_bulk_scaled1 = c_bulk1
        self.c_bulk_scaled2 = c_bulk2
        self.c_surf_scaled1 = c_surf1
        self.c_surf_scaled2 = c_surf2
        self.use_surf_bc = use_surf_bc

# Simulation conditions
Vapp = [-0.2,-0.1,0,0.1,0.2]
concentrations = Concentration_BC(170, 0.6, 0.1, 0.3, 0.6, False)
data = []

for vapp in Vapp:
    data.append(simulator.simulate_mPNP(vapp, 0,concentrations))
    print("Vapp = "+ str(vapp))


# Specify the filename of the .mat file
matfile = 'test_mat.mat'
scipy.io.savemat(matfile, mdict={'out': data}, oned_as='row')

# Now load in the data from the .mat that was just saved
# matdata = scipy.io.loadmat(matfile)



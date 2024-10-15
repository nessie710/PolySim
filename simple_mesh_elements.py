from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import create_mesh_function
import numpy as np

import pyvista
from dolfinx import plot

try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)


def partitionMesh(model):
    # Number of partitions
    N = int(gmsh.onelab.getNumber("Parameters/1Number of partitions")[0])

    # Should we create the boundary representation of the partition entities?
    gmsh.option.setNumber("Mesh.PartitionCreateTopology",
                          gmsh.onelab.getNumber
                          ("Parameters/2Create partition topology (BRep)?")[0])

    # Should we create ghost cells?
    gmsh.option.setNumber("Mesh.PartitionCreateGhostCells",
                          gmsh.onelab.getNumber
                          ("Parameters/3Create ghost cells?")[0])

    # Should we automatically create new physical groups on the partition
    # entities?
    gmsh.option.setNumber("Mesh.PartitionCreatePhysicals",
                          gmsh.onelab.getNumber
                          ("Parameters/3Create new physical groups?")[0])

    # Should we keep backward compatibility with pre-Gmsh 4, e.g. to save the
    # mesh in MSH2 format?
    gmsh.option.setNumber("Mesh.PartitionOldStyleMsh2", 0)

    # Should we save one mesh file per partition?
    gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles",
                          gmsh.onelab.getNumber
                          ("Parameters/4Write one file per partition?")[0])

    if gmsh.onelab.getNumber("Parameters/0Mesh partitioner")[0] == 0:
        # Use Metis to create N partitions
        model.mesh.partition(N)
        # Several options can be set to control Metis: `Mesh.MetisAlgorithm' (1:
        # Recursive, 2: K-way), `Mesh.MetisObjective' (1: min. edge-cut, 2:
        # min. communication volume), `Mesh.PartitionTriWeight' (weight of
        # triangles), `Mesh.PartitionQuadWeight' (weight of quads), ...
    else:
        # Use the `SimplePartition' plugin to create chessboard-like partitions
        gmsh.plugin.setNumber("SimplePartition", "NumSlicesX", N)
        gmsh.plugin.setNumber("SimplePartition", "NumSlicesY", 1)
        gmsh.plugin.setNumber("SimplePartition", "NumSlicesZ", 1)
        
        gmsh.plugin.run("SimplePartition",)


    # Iterate over partitioned entities and print some info (see the first
    # extended tutorial `x1.py' for additional information):
    entities = gmsh.model.getEntities()
    for e in entities:
        partitions = model.getPartitions(e[0], e[1])
        if len(partitions):
            print("Entity " + str(e) + " of type " +
                  model.getType(e[0], e[1]))
            print(" - Partition(s): " + str(partitions))
            print(" - Parent: " + str(model.getParent(e[0], e[1])))
            print(" - Boundary: " + str(model.getBoundary([e])))




# Launch the GUI and handle the "check" event to re-partition the mesh according
# to the choices made in the GUI
def checkForEvent():
    action = gmsh.onelab.getString("ONELAB/Action")
    if len(action) and action[0] == "check":
        gmsh.onelab.setString("ONELAB/Action", [""])
        partitionMesh()
        gmsh.graphics.draw()
    return True

def gmsh_square(model: gmsh.model, name:str, dimensions: int) -> gmsh.model:
    model.add(name)
    model.setCurrent(name)
    square = model.occ.addRectangle(0,0,0,dimensions,dimensions,tag=1)
    model.occ.synchronize()
    model.add_physical_group(dim=2, tags =[square])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
    model.mesh.generate(dim=2)
    return model





def gmsh_simple_1_electrode_domain(model: gmsh.model, name:str, L: float, R_sens):
    """ Create a Gmsh model of a simplified nanoelectrode platform with only one electrode
        for testing purposes

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a domain mesh added.
    """
    gmsh.onelab.set("""[
      {
        "type":"number",
        "name":"Parameters/0Mesh partitioner",
        "values":[0],
        "choices":[0, 1],
        "valueLabels":{"Metis":0, "SimplePartition":1}
      },
      {
        "type":"number",
        "name":"Parameters/1Number of partitions",
        "values":[2],
        "min":1,
        "max":256,
        "step":1
      },
      {
        "type":"number",
        "name":"Parameters/2Create partition topology (BRep)?",
        "values":[0],
        "choices":[0, 1]
      },
      {
        "type":"number",
        "name":"Parameters/3Create ghost cells?",
        "values":[0],
        "choices":[0, 1]
      },
      {
        "type":"number",
        "name":"Parameters/3Create new physical groups?",
        "values":[0],
        "choices":[0, 1]
      },
      {
        "type":"number",
        "name":"Parameters/3Write file to disk?",
        "values":[1],
        "choices":[0, 1]
      },
      {
        "type":"number",
        "name":"Parameters/4Write one file per partition?",
        "values":[0],
        "choices":[0, 1]
      }
    ]""")

    model.add("name")
    model.setCurrent("name")
    gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.PartitionCreateTopology", 1)
    gmsh.option.setNumber("Mesh.PartitionCreatePhysicals", 0)

    
    box = model.occ.addBox(0, 0, 0, L, L, L)

    electrode = model.occ.addDisk(L/2,L/2,0,R_sens,R_sens)
    electrolyte = model.occ.fragment([(3,box)], [(2,electrode)], removeObject=True)
    bulk_marker = 9
    model.occ.synchronize()
    volumes = model.getEntities(dim=3)
    electrolyte_marker = 11
    model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], electrolyte_marker) 
    model.setPhysicalName(volumes[0][0], electrolyte_marker, "Electrolyte volume")

    surfaces = model.getEntities(dim=2)
    electrode_marker = 7
    wall_marker = 5
    walls = []

    for surface in surfaces:
        com = model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [L/2, L/2, L]):
            model.addPhysicalGroup(surface[0], [surface[1]], bulk_marker)
            bulk = surface[1]
            model.setPhysicalName(surface[0], bulk_marker, "Bulk_electrolyte")
        elif np.allclose(com, [L/2, L/2, 0]):     
            for entity in model.occ.getEntitiesInBoundingBox(L/2-R_sens-R_sens/2,L/2-R_sens-R_sens/2,-1e-7,L/2+R_sens+R_sens/2,L/2+R_sens+R_sens/2,1e-7, dim=2):
                if surface[1] == entity[1]:
                    electrode1 = surface[1]
                    model.addPhysicalGroup(surface[0], [surface[1]], electrode_marker)
                    model.setPhysicalName(surface[0], electrode_marker, "Electrode 1")
            
        else:
            walls.append(surface[1])


    model.addPhysicalGroup(2, walls, wall_marker)
    model.setPhysicalName(2, wall_marker, "Walls")
    

    distance = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(distance, "FacesList", [electrode1])


    r = 2
    resolution = r / 10
    threshold = model.mesh.field.add("Threshold")
    model.mesh.field.setNumber(threshold, "IField", distance)
    model.mesh.field.setNumber(threshold, "SizeMin", resolution/2)  # Dimensione minima vicino all'elettrodo
    model.mesh.field.setNumber(threshold, "SizeMax", 40*resolution) 
    # model.mesh.field.setNumber(threshold, "LcMin", resolution/6)
    # model.mesh.field.setNumber(threshold, "LcMax", 40 * resolution)
    model.mesh.field.setNumber(threshold, "DistMin",  r/6)
    model.mesh.field.setNumber(threshold, "DistMax", 10*r)
    minimum = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
    model.mesh.field.setAsBackgroundMesh(minimum)

    model.occ.synchronize()
    #partitionMesh(model)
    model.mesh.generate(dim=3)
    
    if MPI.COMM_WORLD.rank == 0:  
        print("Generated Mesh\n")
    gmsh.write("mesh3D.msh")
    
    
    return (bulk_marker,electrode_marker,wall_marker)


def gmsh_5x5_cylinder(model: gmsh.model, name:str, L: float, R_sens, R_cylinder, L_cylinder, dz, pitch):
    """ Create a Gmsh model of a simplified nanoelectrode platform with only one electrode
        for testing purposes

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a domain mesh added.
    """
    
    model.add(name)
    model.setCurrent(name)
 
    box = model.occ.addBox(0, 0, 0, L, L, L)

    electrode1 = model.occ.addDisk(L/2-2*(2*R_sens+pitch),L/2+2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode2 = model.occ.addDisk(L/2-(2*R_sens+pitch),L/2+2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode3 = model.occ.addDisk(L/2,L/2+2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode4 = model.occ.addDisk(L/2+(2*R_sens+pitch),L/2+2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode5 = model.occ.addDisk(L/2+2*(2*R_sens+pitch),L/2+2*(2*R_sens+pitch),0,R_sens,R_sens)

    electrode6 = model.occ.addDisk(L/2-2*(2*R_sens+pitch),L/2+(2*R_sens+pitch),0,R_sens,R_sens)
    electrode7 = model.occ.addDisk(L/2-(2*R_sens+pitch),L/2+(2*R_sens+pitch),0,R_sens,R_sens)
    electrode8 = model.occ.addDisk(L/2,L/2+(2*R_sens+pitch),0,R_sens,R_sens)
    electrode9 = model.occ.addDisk(L/2+(2*R_sens+pitch),L/2+(2*R_sens+pitch),0,R_sens,R_sens)
    electrode10 = model.occ.addDisk(L/2+2*(2*R_sens+pitch),L/2+(2*R_sens+pitch),0,R_sens,R_sens)

    electrode11 = model.occ.addDisk(L/2-2*(2*R_sens+pitch),L/2,0,R_sens,R_sens)
    electrode12 = model.occ.addDisk(L/2-(2*R_sens+pitch),L/2,0,R_sens,R_sens)
    electrode13 = model.occ.addDisk(L/2,L/2,0,R_sens,R_sens)
    electrode14 = model.occ.addDisk(L/2+(2*R_sens+pitch),L/2,0,R_sens,R_sens)
    electrode15 = model.occ.addDisk(L/2+2*(2*R_sens+pitch),L/2,0,R_sens,R_sens)

    electrode16 = model.occ.addDisk(L/2-2*(2*R_sens+pitch),L/2-(2*R_sens+pitch),0,R_sens,R_sens)
    electrode17 = model.occ.addDisk(L/2-(2*R_sens+pitch),L/2-(2*R_sens+pitch),0,R_sens,R_sens)
    electrode18 = model.occ.addDisk(L/2,L/2-(2*R_sens+pitch),0,R_sens,R_sens)
    electrode19 = model.occ.addDisk(L/2+(2*R_sens+pitch),L/2-(2*R_sens+pitch),0,R_sens,R_sens)
    electrode20 = model.occ.addDisk(L/2+2*(2*R_sens+pitch),L/2-(2*R_sens+pitch),0,R_sens,R_sens)

    electrode21 = model.occ.addDisk(L/2-2*(2*R_sens+pitch),L/2-2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode22 = model.occ.addDisk(L/2-(2*R_sens+pitch),L/2-2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode23 = model.occ.addDisk(L/2,L/2-2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode24 = model.occ.addDisk(L/2+(2*R_sens+pitch),L/2-2*(2*R_sens+pitch),0,R_sens,R_sens)
    electrode25 = model.occ.addDisk(L/2+2*(2*R_sens+pitch),L/2-2*(2*R_sens+pitch),0,R_sens,R_sens)

    #analyte = model.occ.add_cylinder(0,L_cylinder/2,dz,0,1,0,R_cylinder)

    electrolyte = model.occ.fragment([(3,box)], 
                                     [(2,electrode1),(2,electrode2),(2,electrode3),(2,electrode4),(2,electrode5),
                                      (2,electrode6),(2,electrode7),(2,electrode8),(2,electrode9),(2,electrode10),
                                      (2,electrode11),(2,electrode12),(2,electrode13),(2,electrode14),(2,electrode15),
                                      (2,electrode16),(2,electrode17),(2,electrode18),(2,electrode19),(2,electrode20),
                                      (2,electrode21),(2,electrode22),(2,electrode23),(2,electrode24),(2,electrode25)], 
                                     removeObject=True)
    #model.occ.cut([(3,electrolyte)],[(3,analyte)], removeObject=False)

    bulk_marker = 9
    model.occ.synchronize()
    volumes = model.getEntities(dim=3)
    electrolyte_marker = 11
    analyte_marker = 17
    model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], electrolyte_marker) 
    model.setPhysicalName(volumes[0][0], electrolyte_marker, "Electrolyte volume")
    #model.addPhysicalGroup(volumes[1][0], [volumes[1][1]], analyte_marker)
    #model.setPhysicalName(volumes[1][0], analyte_marker, "Analyte volume")

    surfaces = model.getEntities(dim=2)
    electrode_marker = 7
    wall_marker = 5
    central_electrode_marker = 23
    counter_electrodes_marker = 19
    activated_electrode_marker = 21
    walls = []
    activated_electrodes = []
    counter_electrodes = []

    for surface in surfaces:
        com = model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [L/2, L/2, L]):
            model.addPhysicalGroup(surface[0], [surface[1]], bulk_marker)
            bulk = surface[1]
            model.setPhysicalName(surface[0], bulk_marker, "Bulk_electrolyte")
        elif surface in model.occ.getEntitiesInBoundingBox(0,L/2-R_sens-pitch/2,-1e-7,L,L/2+R_sens+pitch/2,1e-7, dim=2):
            activated_electrodes.append(surface[1])
        elif surface in model.occ.getEntitiesInBoundingBox(0,0,-1e-7,L,L/2-R_sens-pitch/2,1e-7, dim=2):
            counter_electrodes.append(surface[1])
        elif surface in model.occ.getEntitiesInBoundingBox(0,L/2+R_sens+pitch/2,-1e-7,L,L,1e-7, dim=2):
            counter_electrodes.append(surface[1])
        else:
            walls.append(surface[1])

        if surface in model.occ.getEntitiesInBoundingBox(L/2-R_sens-pitch/2,L/2-R_sens-pitch/2,-1e-7,L/2+R_sens+pitch/2,L/2+R_sens+pitch/2,1e-7,dim=2):
            model.addPhysicalGroup(surface[0], [surface[1]], central_electrode_marker)
            model.setPhysicalName(surface[0], central_electrode_marker, "Central electrode")


    model.addPhysicalGroup(2, activated_electrodes, activated_electrode_marker)
    model.setPhysicalName(2, activated_electrode_marker, "Activated electrodes")
    
    model.addPhysicalGroup(2, counter_electrodes, counter_electrodes_marker)
    model.setPhysicalName(2, counter_electrodes_marker, "Counter electrodes")

    model.addPhysicalGroup(2, walls, wall_marker)
    model.setPhysicalName(2, wall_marker, "Walls")
    

    distance = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(distance, "FacesList", activated_electrodes + counter_electrodes)

    print("Defined Everything")

    r = 30
    resolution = r / 10
    threshold = model.mesh.field.add("Threshold")
    model.mesh.field.setNumber(threshold, "IField", distance)
    model.mesh.field.setNumber(threshold, "SizeMin", resolution)  # Dimensione minima vicino all'elettrodo
    model.mesh.field.setNumber(threshold, "SizeMax", 400*resolution) 
    #model.mesh.field.setNumber(threshold, "LcMin", resolution/6)
    #model.mesh.field.setNumber(threshold, "LcMax", 40 * resolution)
    model.mesh.field.setNumber(threshold, "DistMin",  r/6)
    model.mesh.field.setNumber(threshold, "DistMax", 10*r)
    minimum = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(minimum, "FieldsList", [threshold])
    model.mesh.field.setAsBackgroundMesh(minimum)

    model.occ.synchronize()
    #partitionMesh(model)
    model.mesh.generate(dim=3)
    
    if MPI.COMM_WORLD.rank == 0:  
        print("Generated Mesh\n")
    gmsh.write("mesh3D_5x5.msh")
    
    
    return (central_electrode_marker, bulk_marker, activated_electrode_marker, counter_electrodes_marker, wall_marker)














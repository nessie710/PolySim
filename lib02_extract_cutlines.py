import numpy as np
from dolfinx import geometry as gm

def extract_central_cutline(domain, L_scaled):
    # EXTRACT CENTRAL CUTLINE
    tol = 0.001
    x = np.linspace(0, L_scaled-tol, 101)
    points = np.zeros((3,101))
    points[2] = x
    points[1] = np.ones((1,101))*L_scaled/2
    points[0] = np.ones((1,101))*L_scaled/2
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
    
    return (points_on_proc, cells)


def extract_central_cutline_1D(domain, L_scaled):
    # EXTRACT CENTRAL CUTLINE
    tol = 0.001
    x = np.linspace(0, L_scaled-tol, 101)
    points = np.zeros((3,101))  # Array di 101 punti con 3 colonne (x, y=0, z=0)
    points[0] = x  # Solo la coordinata x varia
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
    
    return (points_on_proc, cells)
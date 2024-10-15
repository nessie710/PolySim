import dolfinx.fem as fem

def assemble_boundary_conditions_stationary(V, facet_tags, fdim, V_bulk_scaled, Vapp_scaled, c_bulk_scaled, bulk_marker, electrode_marker):
    V_split = V.sub(2)
    V_potential, _ = V_split.collapse()
    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + V_bulk_scaled)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(bulk_marker))
    bc_potential_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    
    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + Vapp_scaled)
    dofs_surface = fem.locate_dofs_topological((V_split,V_potential),fdim,facet_tags.find(electrode_marker))
    bc_potential_surface = fem.dirichletbc(ud, dofs_surface, V_split)
    
    
    V_split = V.sub(0)
    V_c1, _ = V_split.collapse()
    ud = fem.Function(V_c1)
    ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c1), fdim, facet_tags.find(bulk_marker))
    bc_c1_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    
    
    V_split = V.sub(1)
    V_c2, _ = V_split.collapse()
    ud = fem.Function(V_c2)
    ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c2), fdim, facet_tags.find(bulk_marker))
    bc_c2_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)  
    
    bcs = [bc_potential_bulk, bc_potential_surface, bc_c1_bulk, bc_c2_bulk]
    
    return bcs


def assemble_boundary_conditions_AC(VAC, facet_tags, fdim, V_bulk_scaled, Vapp_AC_scaled, c_bulk_scaled, bulk_marker, electrode_marker, wall_marker):
    V_split = VAC.sub(2)
    V_potential, _ = V_split.collapse()
    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 )
    dofs_bulk = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(bulk_marker))
    bc_potential_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    dofs_walls = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(wall_marker))
    bc_potential_walls = fem.dirichletbc(ud, dofs_walls, V_split)


    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + Vapp_AC_scaled)
    dofs_surface = fem.locate_dofs_topological((V_split,V_potential),fdim,facet_tags.find(electrode_marker))
    bc_potential_surface = fem.dirichletbc(ud, dofs_surface, V_split)



    V_split = VAC.sub(0)
    V_c1, _ = V_split.collapse()
    ud = fem.Function(V_c1)
    ud.interpolate(lambda x : x[0]*0)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c1), fdim, facet_tags.find(bulk_marker))
    bc_c1_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    dofs_walls = fem.locate_dofs_topological((V_split, V_c1), fdim, facet_tags.find(wall_marker))
    bc_c1_walls = fem.dirichletbc(ud, dofs_walls, V_split)


    V_split = VAC.sub(1)
    V_c2, _ = V_split.collapse()
    ud = fem.Function(V_c2)
    ud.interpolate(lambda x : x[0]*0 )
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c2), fdim, facet_tags.find(bulk_marker))
    bc_c2_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)  
    dofs_walls = fem.locate_dofs_topological((V_split, V_c2), fdim, facet_tags.find(wall_marker))
    bc_c2_walls = fem.dirichletbc(ud, dofs_walls, V_split)

    bcs = [bc_potential_bulk, bc_potential_surface, bc_c1_walls, bc_c2_walls, bc_potential_walls]
    
    return bcs


def assemble_boundary_conditions_stationary_electrode_matrix(V, facet_tags, fdim, V_bulk_scaled, Vapp_scaled, c_bulk_scaled, bulk_marker, activated_electrode, counter_electrode):
    V_split = V.sub(2)
    V_potential, _ = V_split.collapse()
    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + V_bulk_scaled)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(bulk_marker))
    bc_potential_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)

    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + V_bulk_scaled)
    dofs_counter = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(counter_electrode))
    bc_potential_counter = fem.dirichletbc(ud, dofs_counter, V_split)
    
    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + Vapp_scaled)
    dofs_surface = fem.locate_dofs_topological((V_split,V_potential),fdim,facet_tags.find(activated_electrode))
    bc_potential_activated = fem.dirichletbc(ud, dofs_surface, V_split)
    
    
    V_split = V.sub(0)
    V_c1, _ = V_split.collapse()
    ud = fem.Function(V_c1)
    ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c1), fdim, facet_tags.find(bulk_marker))
    bc_c1_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    
    
    V_split = V.sub(1)
    V_c2, _ = V_split.collapse()
    ud = fem.Function(V_c2)
    ud.interpolate(lambda x : x[0]*0 + c_bulk_scaled)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c2), fdim, facet_tags.find(bulk_marker))
    bc_c2_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)  
    
    bcs = [bc_potential_bulk, bc_potential_activated, bc_potential_counter, bc_c1_bulk, bc_c2_bulk]
    
    return bcs



def assemble_boundary_conditions_AC_electrode_matrix(VAC, facet_tags, fdim, V_bulk_scaled, Vapp_AC_scaled, c_bulk_scaled, bulk_marker, activated_electrode, counter_electrode, wall_marker):
    V_split = VAC.sub(2)
    V_potential, _ = V_split.collapse()
    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 )
    dofs_bulk = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(bulk_marker))
    bc_potential_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    dofs_walls = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(wall_marker))
    bc_potential_walls = fem.dirichletbc(ud, dofs_walls, V_split)
    dofs_counter = fem.locate_dofs_topological((V_split,V_potential), fdim, facet_tags.find(counter_electrode))
    bc_potential_counter = fem.dirichletbc(ud, dofs_counter, V_split)


    ud = fem.Function(V_potential)
    ud.interpolate(lambda x : x[0]*0 + Vapp_AC_scaled)
    dofs_activated = fem.locate_dofs_topological((V_split,V_potential),fdim,facet_tags.find(activated_electrode))
    bc_potential_activated = fem.dirichletbc(ud, dofs_activated, V_split)



    V_split = VAC.sub(0)
    V_c1, _ = V_split.collapse()
    ud = fem.Function(V_c1)
    ud.interpolate(lambda x : x[0]*0)
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c1), fdim, facet_tags.find(bulk_marker))
    bc_c1_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)
    dofs_walls = fem.locate_dofs_topological((V_split, V_c1), fdim, facet_tags.find(wall_marker))
    bc_c1_walls = fem.dirichletbc(ud, dofs_walls, V_split)


    V_split = VAC.sub(1)
    V_c2, _ = V_split.collapse()
    ud = fem.Function(V_c2)
    ud.interpolate(lambda x : x[0]*0 )
    dofs_bulk = fem.locate_dofs_topological((V_split,V_c2), fdim, facet_tags.find(bulk_marker))
    bc_c2_bulk = fem.dirichletbc(ud, dofs_bulk, V_split)  
    dofs_walls = fem.locate_dofs_topological((V_split, V_c2), fdim, facet_tags.find(wall_marker))
    bc_c2_walls = fem.dirichletbc(ud, dofs_walls, V_split)

    bcs = [bc_potential_bulk, bc_potential_activated, bc_potential_counter, bc_c1_walls, bc_c2_walls, bc_potential_walls]
    
    return bcs
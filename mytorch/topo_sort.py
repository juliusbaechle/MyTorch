def build_topo(tensor, visited=set(), topo_order=[]):
    if id(tensor) in visited:
        return topo_order
    
    visited.add(id(tensor))

    # build_topo for each parent
    for parent_ref in getattr(tensor, "_parents", ()):
        parent = parent_ref()
        if parent is not None:
            build_topo(parent, visited, topo_order)

    topo_order.append(tensor)
    return topo_order
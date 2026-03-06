def build_topo(tensor, visited=None, topo_order=None):
    if visited is None:
        visited = set()

    if topo_order is None:
        topo_order = []

    if id(tensor) in visited:
        return topo_order
    
    visited.add(id(tensor))

    for p_ref in tensor._parents:
        build_topo(p_ref(), visited, topo_order)

    topo_order.append(tensor)
    return topo_order
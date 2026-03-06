import re
import networkx as nx
from .tensor import Tensor
import matplotlib.pyplot as plt

def clean_op_name(name: str) -> str:
    """
    All backward methods are named as _{op}_backward
    so we can get the op name from our grad_fn name
    """
    name = re.sub(r"^_+", "", name)
    name = re.sub(r"_backward$", "", name)
    return name.capitalize()

def get_shape(tensor : Tensor):

    """
    We can plot the shapes of the data in the nodes
    so we can grab it like this!
    """
    if hasattr(tensor, "shape"):
        try:
            return str(tuple(tensor.shape))
        except Exception:
            return ""
    return ""

def build_graph(topo_order):

    """
    This is just a plotting function with networkx so we can see
    the result of the forward and backward pass!
    """
    ### Compute our Topological Sort ###
    total_nodes = len(topo_order)

    G = nx.MultiDiGraph()
    id_to_name = {}

    for idx, t in enumerate(topo_order):
        node_id = id(t)
        fwd_num = idx + 1
        bwd_num = total_nodes - idx

        if hasattr(t, "grad_fn") and t.grad_fn is not None:
            op_name = clean_op_name(getattr(t.grad_fn, "__name__", type(t.grad_fn).__name__))
        else:
            op_name = "Param" if t.requires_grad else "Leaf"

        ### Get the metadata ###
        shape_info = f" {get_shape(t)}" if get_shape(t) else ""
        node_name = f"{fwd_num}: {op_name}{shape_info}"

        G.add_node(node_name)
        id_to_name[node_id] = (node_name, bwd_num)

    for t in topo_order:
        node_id = id(t)
        node_name, bwd_num = id_to_name[node_id]

        for parent_ref in getattr(t, "_parents", ()):
            parent = parent_ref()
            parent_id = id(parent)
            parent_name, _ = id_to_name[parent_id]

            # Forward edge (no label)
            G.add_edge(parent_name, node_name, key="fwd", direction="forward")

            # Backward edge (with label)
            G.add_edge(node_name, parent_name, key="bwd", direction="backward",
                       label=str(bwd_num))

    return G

def plot_graph(G):

    import math

    pos = nx.spring_layout(G, k=1 / math.sqrt(len(G.nodes)), iterations=100, seed=42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), facecolor='white')

    ### Forward Pass
    forward_edges = [(u, v) for u, v, data in G.edges(data=True) if 'label' not in data]
    nx.draw_networkx_nodes(G, pos, ax=ax1,
                           node_color='#A3BEF2',
                           node_size=10000,
                           edgecolors='navy',
                           linewidths=3.5)

    nx.draw_networkx_edges(G, pos, ax=ax1,
                           edgelist=forward_edges,
                           arrowstyle='->',            
                           arrowsize=35,               
                           edge_color='#1976D2',
                           width=4,
                           connectionstyle='arc3,rad=0.15',
                           node_size=10000)             

    nx.draw_networkx_labels(G, pos, ax=ax1,
                            font_size=17,
                            font_weight='bold',
                            font_family='DejaVu Sans',
                            bbox=dict(facecolor='white', edgecolor='none', pad=5, alpha=0.9))

    ax1.set_title("Forward Pass", fontsize=24, fontweight='bold', pad=30, color='#0D47A1')
    ax1.axis('off')
    ax1.margins(0.2)

    ### Backward Pass ###
    backward_edges = [(u, v) for u, v, data in G.edges(data=True) if 'label' in data]

    nx.draw_networkx_nodes(G, pos, ax=ax2,
                           node_color='#FF9999',
                           node_size=10000,
                           edgecolors='darkred',
                           linewidths=3.5)

    nx.draw_networkx_edges(G, pos, ax=ax2,
                           edgelist=backward_edges,
                           style='--',
                           arrowstyle='->',
                           arrowsize=35,
                           edge_color='#C62828',
                           width=4,
                           connectionstyle='arc3,rad=-0.15',
                           node_size=10000)

    nx.draw_networkx_labels(G, pos, ax=ax2,
                            font_size=17,
                            font_weight='bold',
                            font_family='DejaVu Sans',
                            bbox=dict(facecolor='white', edgecolor='none', pad=5, alpha=0.9))

    # Backward edge labels (numbers)
    backward_labels = {(u, v): data['label'] for u, v, data in G.edges(data=True) if 'label' in data}
    nx.draw_networkx_edge_labels(G, pos, ax=ax2,
                                 edge_labels=backward_labels,
                                 font_size=14,
                                 font_color='#B71C1C',
                                 font_weight='bold',
                                 label_pos=0.6,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax2.set_title("Backward Pass", fontsize=24, fontweight='bold', pad=30, color='#B71C1C')
    ax2.axis('off')
    ax2.margins(0.2)

    plt.suptitle("Computation Graph: Forward & Backward Pass", 
                 fontsize=28, fontweight='bold', y=0.98, color='#1A1A1A')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.tight_layout()
    plt.show()

def print_topo(topo : list):
    ids = [id(t) for t in topo]
    for i, t in enumerate(topo):
        if t.grad_fn:
            name = clean_op_name(t.grad_fn.__name__)
        else:
            name = "Param" if t.requires_grad else "Leaf"
        
        parents = []
        for p in t._parents:
            index = ids.index(id(p()))
            parents.append(index)
        print(f"{i}: {name} {t.shape} {parents}")
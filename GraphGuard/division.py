import numpy as np
import networkx as nx
import hashlib
import torch
from copy import deepcopy, copy

DEFAULT=-99

# Define feature functions
def get_node_tags(graph):
    return np.array(graph.node_tags)

def get_degree_centrality(graph):
    return nx.degree_centrality(graph.g).values()

def get_betweenness_centrality(graph):
    return nx.betweenness_centrality(graph.g).values()

def get_closeness_centrality(graph):
    return nx.closeness_centrality(graph.g).values()

def get_eigenvector_centrality(graph):
    return nx.eigenvector_centrality(graph.g).values()

def get_node_id(graph):
    return np.arange(len(graph.g))

# Map feature method to corresponding function
features_func_map = {
    "tags": get_node_tags,
    "degree": get_degree_centrality,
    "betweenness": get_betweenness_centrality,
    "closeness": get_closeness_centrality,
    "eigenvector": get_eigenvector_centrality,
    "id": get_node_id
}

# Define hash functions
def md5_hash(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)

def sha256_hash(x):
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

def sha512_hash(x):
    return int(hashlib.sha512(str(x).encode()).hexdigest(), 16)

# Map hash method to corresponding function
hash_func_map = {
    "hash": hash,
    "md5": md5_hash,
    "sha256": sha256_hash,
    "sha512": sha512_hash
}

def get_subgraph_drop_edges(graph, edges_to_remove):
    """
    Get a subgraph by removing edges from a given graph

    Parameters:
    graph (SV2Graph): The input graph
    edges_to_remove (list): List of edges to remove from the input graph

    Returns:
    subgraph (SV2Graph): The subgraph obtained by removing the specified edges
    """
    subgraph = copy(graph) # deepcopy(graph)
    subgraph.g = graph.g.copy()
    # key code: removing the specified edges
    subgraph.g.remove_edges_from(edges_to_remove)
    edge_matrix = list(subgraph.g.edges)
    edge_matrix.extend([[i, j] for j, i in edge_matrix])
    subgraph.edge_mat = torch.tensor(edge_matrix, dtype=int).reshape(2,-1)
    subgraph.neighbors = [list(subgraph.g.neighbors(node)) for node in range(len(subgraph.g))]
    subgraph.max_neighbor = max([len(neighbors) for neighbors in subgraph.neighbors])

    return subgraph

def get_edge_hash(graph, hash_func, features):
    """
    Compute hash values for the edges in a given graph

    Parameters:
    graph (networkx.Graph): The input graph
    hash_func (function): The hash function to use
    features (dict): A dictionary mapping node IDs to their feature vectors

    Returns:
    edge_hash (list): A list of hash values for the edges in the graph
    """

    node_categories = [f"{features[start_node]};{features[end_node]}" for start_node, end_node in graph.edges]
    edge_hash = [hash_func(category) for category in node_categories]

    return edge_hash

def get_node_hash(graph, hash_func, features):
    """
    Compute hash values for the NODEs in a given graph

    Parameters:
    graph (networkx.Graph): The input graph
    hash_func (function): The hash function to use
    features (dict): A dictionary mapping node IDs to their feature vectors

    Returns:
    node_hash (list): A list of hash values for the NODEs in the graph
    """
    node_hash = [hash_func(feature) for feature in features]

    return node_hash

def graph_structure_division(graph, args):
    """
    Divide a graph into subgraphs based on their edge hash values

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    
    # Compute the features and edge hash values for the graph
    features = args.features_func(graph)
    edge_hash = get_edge_hash(graph.g, args.hash_func, features)
    # Divide the graph based on the edge hash values
    group = np.array([h % args.num_group for h in edge_hash])
    # Create subgraphs for each group
    subgraphs = []
    G_edges = np.array(list(graph.g.edges))
    for i in range(args.num_group):
        edges_to_remove = G_edges[group != i]
        subgraph = get_subgraph_drop_edges(graph, edges_to_remove)
        subgraphs.append(subgraph)

    return subgraphs

def graph_feature_division(graph, args):
    """
    Divide a graph into subgraphs based on their NODE hash values

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    # Compute the features and NODE hash values for the graph
    features = args.features_func(graph)
    node_hash = get_node_hash(graph.g, args.hash_func, features)
    # Divide the graph based on the NODE hash values
    group = np.array([h % args.num_group for h in node_hash])
    # Create subgraphs for each group
    subgraphs = []
    for i in range(args.num_group):
        subgraph = copy(graph)
        subgraph.node_features[group!=i] = DEFAULT
        subgraphs.append(subgraph)
    return subgraphs

def get_subgraph(graph, idx):
    subgraph = copy(graph) # deepcopy(graph)
    subgraph.g = graph.g.subgraph(idx)
    # key code: removing the specified edges
    edge_matrix = list(subgraph.g.edges)
    edge_matrix.extend([[i, j] for j, i in edge_matrix])
    subgraph.node_features = graph.node_features[idx]
    subgraph.node_tags = [graph.node_tags[i] for i in idx]
    subgraph.edge_mat = torch.tensor(edge_matrix, dtype=int).reshape(2,-1)
    subgraph.neighbors = [list(subgraph.g.neighbors(node)) for node in subgraph.g.nodes]
    if subgraph.neighbors:
        subgraph.max_neighbor = max([len(neighbors) for neighbors in subgraph.neighbors])
    else:
        subgraph.max_neighbor = 0
    if subgraph.max_neighbor is None:
        subgraph.max_neighbor = 0
    return subgraph

def graph_node_division(graph, args):
    """
    Divide a graph into subgraphs based on their NODE hash values

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    # Compute the features and NODE hash values for the graph
    features = args.features_func(graph)
    node_hash = get_node_hash(graph.g, args.hash_func, features)
    # Divide the graph based on the NODE hash values
    group = np.array([h % args.num_group for h in node_hash])
    # Create subgraphs for each group
    subgraphs = []
    for i in range(args.num_group):
        subgraph = get_subgraph(graph, np.nonzero(group==i)[0].tolist())
        subgraphs.append(subgraph)
    return subgraphs


def graph_feature_structure_division(graph, args):
    """
    Divide a graph into subgraphs based on their edge & node hash values

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    # Preserve the original features, since edge_division may affect them (e.g., degree).
    features = args.features_func(graph)
    node_hash = get_node_hash(graph.g, args.hash_func, features)
    group = np.array([h % args.num_group for h in node_hash])
    # Divide the graph based on its structure.
    graphs = graph_structure_division(graph, args)
    # Divide each subgraph based on features.
    subgraphs = []
    for graph in graphs:
        for i in range(args.num_group):
            subgraph = copy(graph)
            subgraph.node_features[group != i] = DEFAULT
            subgraphs.append(subgraph)
    
    return subgraphs

def graph_edge_blank(graph, args):
    """
    Divide a graph into subgraphs without edge

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    graph = get_subgraph_drop_edges(graph, np.array(list(graph.g.edges)))
    return [graph] * args.num_group

def graph_node_blank(graph, args):
    """
    Divide a graph into subgraphs without edge

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    subgraph = deepcopy(graph)
    subgraph.node_features[:] = DEFAULT
    return [subgraph] * args.num_group

def graph_blank(graph, args):
    """
    Divide a graph into subgraphs without edge

    Parameters:
    graph (SV2Graph): The input graph
    args (argparse.Namespace): A namespace containing the required arguments

    Returns:
    subgraphs (list): A list of subgraphs obtained by dividing the input graph
    """
    graph = deepcopy(graph)
    graph.node_features[:] = -1
    graph = get_subgraph_drop_edges(graph, np.array(list(graph.g.edges)))
    return [graph] * args.num_group

# Map division method to corresponding function
division_func_map = {
    "node": graph_node_division,
    "feature": graph_feature_division,
    "structure": graph_structure_division,
    "all": graph_feature_structure_division,
    "no_edge": graph_edge_blank,
    "no_node": graph_node_blank,
    "nothing": graph_blank
}
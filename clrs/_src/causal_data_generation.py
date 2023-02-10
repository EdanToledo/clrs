"""Unit tests for `all add causal functions`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
import jax
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

# Currently hardcoding the rng
_rng = np.random.RandomState(123456)

def create_nx_graph(adjacency_matrix, weighted_matrix):
    """Create a networkx digraph.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
        weighted_matrix (np.ndarray): The adjacency matrix of the graph 
            except instead of 0s and 1s, we use the weight values of the edges.

    Returns:
        nx.Digraph: directed networkx graph
    """
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(
        rows.tolist(), cols.tolist(), np.round(weighted_matrix[rows, cols], 3).tolist()
    )
    gr = nx.DiGraph()
    gr.add_weighted_edges_from(edges)
    return gr

def visualise_graph(adjacency_matrix, weighted_matrix, node_labels=None):
    """Visualise a directed weighted graph.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
        weighted_matrix (np.ndarray): The adjacency matrix of the graph 
            except instead of 0s and 1s, we use the weight values of the edges.
        node_labels (Dict[int, str], optional): Labels to use for the nodes. Defaults to None. If None then the node ids are simply used.
    """
   
    node_list = np.arange(len(adjacency_matrix)).tolist()

    gr = create_nx_graph(adjacency_matrix, weighted_matrix)

    pos = nx.shell_layout(gr)
    nx.draw(
        gr,
        pos,
        node_size=550,
        nodelist=node_list,
        with_labels=False,
    )
    edge_labels = nx.get_edge_attributes(gr, "weight")
    nx.draw_networkx_edge_labels(gr, pos, edge_labels, font_size=8)
    if node_labels:
        labels = {}
        for node in gr.nodes().keys():
            labels[node] = node_labels[node]
    
        nx.draw_networkx_labels(gr, pos, labels, font_size=6, font_color="black")

    plt.show()


def _random_causal_graph(nb_nodes, p=0.5, low=0.0, high=1.0):
    """_summary_

    Args:
        nb_nodes (int): number of nodes
        p (float, optional): _description_. Defaults to 0.5.
        low (float, optional): _description_. Defaults to 0.0.
        high (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """

    # Generate graphs until it is a fully connected graph
    while True:
        # Create a weighted directed acyclic graph
        mat = _rng.binomial(1, p, size=(nb_nodes, nb_nodes))
        mat = np.triu(mat, k=1)
        perm = _rng.permutation(nb_nodes)  # To allow nontrivial solutions
        mat = mat[perm, :][:, perm]
        adjacency_mat = np.copy(mat)
        weights = _rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
        weighted_mat = mat.astype(float) * weights

        # Check if the graph is weakly connected
        if nx.is_weakly_connected(create_nx_graph(adjacency_mat, weighted_mat)):
            # Then check if any node is left out i.e there in-degree == out-degree == 0
            # Get indices of nodes that have the same in-out-degree
            indices = np.where(np.sum(adjacency_mat, axis=0) == np.sum(adjacency_mat, axis=1))
            # Check if any of those nodes have their degree == 0
            if not np.any(np.sum(adjacency_mat, axis=0)[indices] == 0):
                break
            
    # Find node indices that are exogenous variables
    exogenous_nodes = np.where(np.sum(adjacency_mat, axis=0) == 0)
    # Find node indices that are endogenous variables
    endogenous_nodes = np.delete(np.arange(nb_nodes), exogenous_nodes)

    # Choose the number of data points - I believe this will be one in the way clrs works but could be wrong - coded generally
    data_points = 1

    # Create the initial data that occupies each node space
    node_data = np.zeros((nb_nodes, data_points))

    # For every exogenous node - set its initial values - currently this needs to change - not sure what to put here
    for exo_node in exogenous_nodes:
        node_data[exo_node] = _rng.randint(1, 2, size=(data_points,))

    # For every endogenous node - run the calculations through the
    # DAG - currently uses the weights of the DAG to represent the
    # functional relationship and simply performing linear calculations.
    # Node values join together either by sum or product which is chosen by random.
    operations = [np.sum, np.prod]
    
    # Recursively set all node values
    def set_value(adjacency_matrix, weighted_matrix, endo_node, node_data):
        if np.all(node_data[endo_node] != 0):
            return node_data

        parent_nodes = np.where(adjacency_mat[:, endo_node] == 1)
        if len(parent_nodes[0])>0:
            parent_nodes_values = node_data[parent_nodes]
            parent_node_fill_indices = parent_nodes[0][np.where(parent_nodes_values == 0)[0]]
            for parent_node_id in parent_node_fill_indices:
                node_data = set_value(adjacency_matrix, weighted_matrix, parent_node_id, node_data)
        
            endo_node_value = node_data[parent_nodes] * np.expand_dims(
                np.squeeze(weighted_matrix[parent_nodes, endo_node]), -1
            )
            op_index = _rng.choice(len(operations))
            op = operations[op_index]
            endo_node_value = op(endo_node_value, axis=0)
            node_data[endo_node] = endo_node_value

        return node_data

    for endo_node in endogenous_nodes:
        node_data = set_value(adjacency_mat, weighted_mat, endo_node, node_data)


    return adjacency_mat, weighted_mat, exogenous_nodes, endogenous_nodes, node_data


def test_causal_data_generation(nb_nodes, low=0.0, high=1.0, p=(0.5,)):
    (
        adjacency_mat,
        weighted_mat,
        exogenous_nodes,
        endogenous_nodes,
        node_data,
    ) = _random_causal_graph(
        nb_nodes=nb_nodes,
        p=_rng.choice(p),
        low=low,
        high=high,
    )

    visualise_graph(adjacency_mat, weighted_mat, node_labels = np.round(np.squeeze(node_data),2).tolist())

    

if __name__ == "__main__":
    test_causal_data_generation(5, 5)

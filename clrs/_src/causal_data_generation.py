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
_rng = np.random.RandomState(42)

def visualise_graph(adjacency_matrix, weighted_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(
        rows.tolist(), cols.tolist(), np.round(weighted_matrix[rows, cols], 3).tolist()
    )
    gr = nx.DiGraph()
    gr.add_weighted_edges_from(edges)
    pos = nx.spring_layout(gr)
    nx.draw(
        gr,
        pos,
        node_size=500,
        nodelist=np.arange(len(adjacency_matrix)).tolist(),
        with_labels=True,
    )
    edge_labels = nx.get_edge_attributes(gr, "weight")
    nx.draw_networkx_edge_labels(gr, pos, edge_labels)
    plt.show()


def _random_causal_graph(nb_nodes, p=0.5, low=0.0, high=1.0):
    """Random causal graph and data"""

    # Create a weighted directed acyclic graph
    mat = _rng.binomial(1, p, size=(nb_nodes, nb_nodes))
    mat = np.triu(mat, k=1)
    p = _rng.permutation(nb_nodes)  # To allow nontrivial solutions
    mat = mat[p, :][:, p]
    adjacency_mat = np.copy(mat)
    weights = _rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
    weighted_mat = mat.astype(float) * weights

    # Find node indices that are exogenous variables
    exogenous_nodes = np.where(np.sum(mat, axis=0) == 0)
    # Find node indices that are endogenous variables
    endogenous_nodes = np.delete(np.arange(nb_nodes), exogenous_nodes)

    # Choose the number of data points - I believe this will be one in the way clrs works but could be wrong - coded generally
    data_points = 1

    # Create the initial data that occupies each node space
    node_data = np.zeros((nb_nodes, data_points))

    # For every exogenous node - set its initial values - currently this needs to change - not sure what to put here
    for exo_node in exogenous_nodes:
        node_data[exo_node] = np.random.randint(0, 100, size=(data_points,))

    # For every endogenous node - run the calculations through the
    # DAG - currently uses the weights of the DAG to represent the
    # functional relationship and simply performing linear calculations like an MLP - can change
    for endo_node in endogenous_nodes:
        parent_nodes = np.where(adjacency_mat[:, endo_node] == 1)
        endo_node_value = node_data[parent_nodes] * np.expand_dims(
            np.squeeze(weighted_mat[parent_nodes, endo_node]), -1
        )
        endo_node_value = np.sum(endo_node_value, axis=0)
        node_data[endo_node] = endo_node_value

    return adjacency_mat, weighted_mat, exogenous_nodes, endogenous_nodes, node_data


def test_causal_data_generation(nb_nodes, low=0.0, high=1.0, p=(0.5,)):
    (
        adjacency_mat,
        weighted_mat,
        exogenous_nodes,
        endogenous_nodes,
    ) = _random_causal_graph(
        nb_nodes=nb_nodes,
        p=_rng.choice(p),
        directed=True,
        acyclic=True,
        weighted=True,
        low=low,
        high=high,
    )

    visualise_graph(adjacency_mat, weighted_mat)



if __name__ == "__main__":
    test_causal_data_generation(5, 5)

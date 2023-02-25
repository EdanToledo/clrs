import numpy as np

import matplotlib.pyplot as plt
from causalgraphicalmodels import CausalGraphicalModel
from causalgraphicalmodels import StructuralCausalModel
import networkx as nx
import seaborn as sns


# Currently hardcoding the rng
_rng = np.random.RandomState(5)


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


def _random_causal_graph(nb_nodes, p=0.5, low=0.0, high=1.0, noise=0.1, binomial_exogenous_variables = True, binomial_probability = 0.6):
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
            indices = np.where(
                np.sum(adjacency_mat, axis=0) == np.sum(adjacency_mat, axis=1)
            )
            # Check if any of those nodes have their degree == 0
            if not np.any(np.sum(adjacency_mat, axis=0)[indices] == 0):
                break

    # Find node indices that are exogenous variables - no incoming connections
    exogenous_nodes = np.where(np.sum(adjacency_mat, axis=0) == 0)
    # Find node indices that are endogenous variables - have incoming connections
    endogenous_nodes = np.delete(np.arange(nb_nodes), exogenous_nodes)

    parent_nodes = [
        np.where(adjacency_mat[:, endo_node] == 1)[0] for endo_node in endogenous_nodes
    ]

    def convert_to_function_sig(parent_nodes):
        function_sig = []
        for parents in parent_nodes:
            var = ""
            for parent in parents:
                var += f"x{parent},"
            function_sig.append(var)

        return function_sig

    def convert_to_causal_relation(parent_nodes, endogenous_nodes):
        math_funs = []
        for endo_node, parents in zip(endogenous_nodes, parent_nodes):
            terms = []
            for parent in parents:
                term = f"{weighted_mat[parent, endo_node]}*x{parent}"
                terms.append(term)

            math_funs.append("+".join(terms))

        return math_funs

    function_signatures = convert_to_function_sig(parent_nodes)
    math_funs = convert_to_causal_relation(parent_nodes, endogenous_nodes)

    exogenous_nodes_scm = {
        f"x{i}": lambda n_samples: np.random.binomial(n=1, p=binomial_probability, size=n_samples) if binomial_exogenous_variables else np.random.uniform(low, high, size=n_samples)
        for i in exogenous_nodes[0]
    }

    endogenous_nodes_scm = {
        f"x{endogenous_nodes[i]}": eval(
            f"lambda {function_signatures[i]} n_samples : np.random.normal(loc={math_funs[i]}, scale={noise})"
        )
        for i in range(len(endogenous_nodes))
    }

    scm = StructuralCausalModel({**exogenous_nodes_scm, **endogenous_nodes_scm})

    return adjacency_mat, weighted_mat, exogenous_nodes, endogenous_nodes, scm


def test_causal_data_generation(nb_nodes, low=0.0, high=1.0, p=[0.1, 0.2, 0.3, 0.5]):
    (
        adjacency_mat,
        weighted_mat,
        exogenous_nodes,
        endogenous_nodes,
        scm,
    ) = _random_causal_graph(
        nb_nodes=nb_nodes,
        p=_rng.choice(p),
        low=low,
        high=high,
        binomial_exogenous_variables=False,
        binomial_probability=0.6
    )

    ds = scm.sample(100)
    print(ds.head())
    scm.cgm.draw().view()


if __name__ == "__main__":
    test_causal_data_generation(10)

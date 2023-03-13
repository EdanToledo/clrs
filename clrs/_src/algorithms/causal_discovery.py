# modified from IC* implementation in causality module
# https://github.com/akelleh/causality/blob/master/causality/inference/search/__init__.py

from typing import Tuple

import chex

# # code for autoreloading modules in ipython
# %load_ext autoreload
# %autoreload 2

import clrs._src.probing as probing
import clrs._src.specs as specs

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time 

import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import itertools

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest
from clrs._src.causal_data_generation import _random_causal_graph

# SZ comments: imports are a lil whack but the code works rn


_Array = np.ndarray
_DataFrame = pd.DataFrame
_Out = Tuple[_Array, _Array, probing.ProbesDict]
_OutputClass = specs.OutputClass


def ic_star(X_df: _DataFrame) -> _Out:
    """IC* algorithm using PC alg in step 1
    X is the NUM_SAMPLES x NUM_VARS array of observations derived from an SCM
    """
    
    chex.assert_rank(X_df.to_numpy(), 2)

    NUM_VARS = X_df.shape[1]
    VAR_NAMES = list(
        X_df.columns.sort_values()
    )  # NOTE: this is sorted for consistency in adjacency matrix and arrows_mat representation
    

    probes = probing.initialize(specs.SPECS["ic_star"])
    
    input_data = np.copy(X_df) # Shape is Num Data points x NUM_VARS
    input_data = np.swapaxes(input_data, 0, 1)
   
    probing.push(probes, specs.Stage.INPUT, next_probe={"X": input_data})

    # Step 1: find undirected graph with conditionally independent vars unconnected
    # initialize completely connected undirected graph
    g = nx.complete_graph(VAR_NAMES)

    # initialize arrows_mat
    # arrows_mat[i,j] == 0 -> no information
    # arrows_mat[i,j] == 1 and arrows_mat[j,i] == 1 -> bidirected edge as defined in Pearl 2000
    # arrows_mat[i,j] == 1 -> directed edge as defined in Pearl 2000
    # arrows_mat[i,j] == 2 -> marked directed edge as defined in Pearl 2000
    arrows_mat = np.zeros([NUM_VARS, NUM_VARS])

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            "node_1": np.zeros(NUM_VARS),
            "node_2": np.zeros(NUM_VARS),
            "node_3": np.zeros(NUM_VARS),
            "S_12": np.zeros(NUM_VARS),
            "A_h": probing.graph(np.copy(nx.to_numpy_array(g))),
            "arrows_h": probing.arrows_probe(np.copy(arrows_mat),3),
        },
    )

    # presets
    independence_test = RobustRegressionTest  # just using robust regression test rn, TODO: check what this does
    max_k = NUM_VARS + 1  # TODO: check why we're adding 1 here
    alpha = 0.05  # TODO: check role of alpha in independence_test

    # find skeleton undirected graph (removing edges between nodes that are conditionally independent)
    # """
    #   For each pair of nodes, run a conditional independence test over
    #   larger and larger conditioning sets to try to find a set that
    #   d-separates the pair.  If such a set exists, cut the edge between
    #   the nodes.  If not, keep the edge.
    # """
    separating_sets = {}

    for N in range(max_k + 1):
        for a, b in list(g.edges()):  # iterate over all pairs of nodes
            a_neighbors = list(g.neighbors(a))
            b_neighbors = list(g.neighbors(b))
            c_candidates = list(set(a_neighbors + b_neighbors) - set([a, b]))
            for S_ab in itertools.combinations(
                c_candidates, N
            ):  # iterate over subsets of size N
                test = independence_test([b], [a], list(S_ab), X_df, alpha)
                if test.independent():
                    # if a and b are conditionally independent given c, remove edge
                    g.remove_edge(a, b)
                    separating_sets[(a, b)] = S_ab

                    
                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "node_1": probing.mask_one(VAR_NAMES.index(a), NUM_VARS),
                            "node_2": probing.mask_one(VAR_NAMES.index(b), NUM_VARS),
                            "node_3": np.zeros(NUM_VARS),
                            "S_12": probing.mask_set(
                                [VAR_NAMES.index(node) for node in list(S_ab)], NUM_VARS
                            ),
                            "A_h": probing.graph(np.copy(nx.to_numpy_array(g))),
                            "arrows_h": probing.arrows_probe(np.copy(arrows_mat),3),
                        },
                    )
                    break
    # pos = nx.planar_layout(g)
    # nx.draw(g, pos, with_labels=True)
    # plt.show()

    # Step 2: orient colliders
    for c in g.nodes():
        for a, b in itertools.combinations(g.neighbors(c), 2):
            if not g.has_edge(a, b):
                S_ab = return_S_ab(separating_sets, a, b)
                if S_ab != None and c not in S_ab:
                    a_idx = VAR_NAMES.index(a)
                    b_idx = VAR_NAMES.index(b)
                    c_idx = VAR_NAMES.index(c)
                    # print(a, a_idx)
                    # print(b, b_idx)
                    # print(c, c_idx)
                    # break

                    arrows_mat[a_idx, c_idx] = 1
                    arrows_mat[b_idx, c_idx] = 1

                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "node_1": probing.mask_one(a_idx, NUM_VARS),
                            "node_2": probing.mask_one(c_idx, NUM_VARS),
                            "node_3": probing.mask_one(
                                b_idx, NUM_VARS
                            ),  # TODO: check if this makes sense
                            "S_12": probing.mask_set(
                                [VAR_NAMES.index(node) for node in list(S_ab)], NUM_VARS
                            ),
                            "A_h": probing.graph(np.copy(nx.to_numpy_array(g))),
                            "arrows_h": probing.arrows_probe(np.copy(arrows_mat),3),
                        },
                    )

    # Step 3: recursively adding as many arrows + markings as possible
    # Follows recursive rules 1 and 2 as detailed in Pearl 2000
    added_arrows = True
    while added_arrows:
        R1_added_arrows = apply_recursion_rule_1(
            g, probes, arrows_mat, VAR_NAMES, NUM_VARS
        )
        R2_added_arrows = apply_recursion_rule_2(
            g, probes, arrows_mat, VAR_NAMES, NUM_VARS
        )
        added_arrows = R1_added_arrows or R2_added_arrows

    A = nx.to_numpy_array(g)

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            "arrows": probing.arrows_probe(np.copy(arrows_mat), 3),
            "A": probing.graph(np.copy(A)),
        },
    )

    probing.finalize(probes)

    return (A, arrows_mat), probes


# ---- helper functions --
def return_S_ab(separating_sets, a, b):
    # returns the separating set of a and b
    # ie, set S_ab such that a and b are conditionally independent given S_ab
    if (a, b) in separating_sets:
        return separating_sets[(a, b)]
    elif (b, a) in separating_sets:
        return separating_sets[(b, a)]
    else:
        return None


def apply_recursion_rule_1(g, probes, arrows_mat, VAR_NAMES, NUM_VARS):
    # R1 as detailed in IC* implementation in Pearl 2000
    added_arrows = False
    for c in g.nodes():
        for a, b in itertools.combinations(g.neighbors(c), 2):
            a_idx = VAR_NAMES.index(a)
            b_idx = VAR_NAMES.index(b)
            c_idx = VAR_NAMES.index(c)

            if not g.has_edge(a, b):
                if (
                    (arrows_mat[a_idx, c_idx] >= 1)
                    and (arrows_mat[b_idx, c_idx] == 0)
                    and not (arrows_mat[c_idx, b_idx] == 2)
                ):
                    arrows_mat[c_idx, b_idx] = 2
                    # g[b][c]['marked'] = True # TODO: check if we want to mark this with star even if arrows_mat
                    added_arrows = True

                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "node_1": probing.mask_one(VAR_NAMES.index(c), NUM_VARS),
                            "node_2": probing.mask_one(VAR_NAMES.index(b), NUM_VARS),
                            "node_3": probing.mask_one(
                                VAR_NAMES.index(a), NUM_VARS
                            ),  # TODO: check if this is the right ordering
                            "S_12": np.zeros(NUM_VARS),
                            "A_h": np.copy(nx.to_numpy_array(g)),
                            "arrows_h": np.copy(arrows_mat),
                        },
                    )
                if (
                    (arrows_mat[b_idx, c_idx] >= 1)
                    and (arrows_mat[a_idx, c_idx] == 0)
                    and not (arrows_mat[c_idx, a_idx] == 2)
                ):
                    arrows_mat[c_idx, a_idx] = 2
                    # g[a][c]['marked'] = True
                    added_arrows = True

                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "node_1": probing.mask_one(VAR_NAMES.index(c), NUM_VARS),
                            "node_2": probing.mask_one(VAR_NAMES.index(a), NUM_VARS),
                            "node_3": probing.mask_one(
                                VAR_NAMES.index(b), NUM_VARS
                            ),  # TODO: check if this is the right ordering
                            "S_12": np.zeros(NUM_VARS),
                            "A_h": np.copy(nx.to_numpy_array(g)),
                            "arrows_h": np.copy(arrows_mat),
                        },
                    )
    return added_arrows


def apply_recursion_rule_2(g, probes, arrows_mat, VAR_NAMES, NUM_VARS):
    # R2 as detailed in IC* implementation in Pearl 2000
    added_arrows = False
    for a, b in g.edges():
        a_idx = VAR_NAMES.index(a)
        b_idx = VAR_NAMES.index(b)

        if arrows_mat[a_idx, b_idx] == 0:
            if marked_directed_path(g, a, b, arrows_mat, VAR_NAMES):
                arrows_mat[a_idx, b_idx] = 1
                added_arrows = True

                probing.push(
                    probes,
                    specs.Stage.HINT,
                    next_probe={
                        "node_1": probing.mask_one(VAR_NAMES.index(a), NUM_VARS),
                        "node_2": probing.mask_one(VAR_NAMES.index(b), NUM_VARS),
                        "node_3": np.zeros(NUM_VARS),
                        "S_12": np.zeros(NUM_VARS),
                        "A_h": np.copy(nx.to_numpy_array(g)),
                        "arrows_h": np.copy(arrows_mat),
                    },
                )
        if arrows_mat[b_idx, a_idx] == 0:
            if marked_directed_path(g, b, a, arrows_mat, VAR_NAMES):
                arrows_mat[b_idx, a_idx] = 1
                added_arrows = True

                probing.push(
                    probes,
                    specs.Stage.HINT,
                    next_probe={
                        "node_1": probing.mask_one(VAR_NAMES.index(b), NUM_VARS),
                        "node_2": probing.mask_one(VAR_NAMES.index(a), NUM_VARS),
                        "node_3": np.zeros(NUM_VARS),
                        "S_12": np.zeros(NUM_VARS),
                        "A_h": np.copy(nx.to_numpy_array(g)),
                        "arrows_h": np.copy(arrows_mat),
                    },
                )
    return added_arrows


def marked_directed_path(g, a, b, arrows_mat, VAR_NAMES):
    # return True if there is a directed path from a to b in g, False otherwise
    seen = [a]
    neighbors = [(a, neighbor) for neighbor in g.neighbors(a)]
    while neighbors:
        (parent, child) = neighbors.pop()
        parent_idx = VAR_NAMES.index(parent)
        child_idx = VAR_NAMES.index(child)
        if arrows_mat[parent_idx, child_idx] == 2:
            if child == b:
                return True
            if child not in seen:
                neighbors += [(child, neighbor) for neighbor in g.neighbors(child)]
            seen.append(child)
    return False


if __name__ == "__main__":
    # for testing example 1
    print("Example 1: Example from Pearl 2000")
    SIZE = 2000
    x0 = np.random.normal(size=SIZE)
    x1 = x0 + np.random.normal(size=SIZE)
    x2 = x0 + np.random.normal(size=SIZE)
    x3 = x1 + x2 + np.random.normal(size=SIZE)
    x4 = x3 + np.random.normal(size=SIZE)
    X = np.transpose(np.vstack((x0, x1, x2, x3, x4)))
    X_df = pd.DataFrame(data=X, columns=[f"x{i}" for i in range(5)])

    (adj_mat, arrows_mat), probes = ic_star(X_df)
    print(f"Variables: {X_df.columns.sort_values()}")
    print(f"Adjacency matrix:\n{adj_mat}")
    print(f"Arrows matrix:\n{arrows_mat}")

    # TODO: take the above and convert it to a nx graph

    # compare with IC algorithm <- should be the same
    variable_types = {f"x{i}": "c" for i in range(X_df.shape[1])}
    graph_comp = IC(RobustRegressionTest).search(X_df, variable_types)
    for i, j in graph_comp.edges():
        print(i, j, graph_comp.get_edge_data(i, j))

    # pos = nx.planar_layout(graph_comp)
    # nx.draw(graph_comp, pos, with_labels=True)
    # plt.show()

    print("####################")
    ##################
    # testing example 2:
    print("Example 2: Randomly generated graph")
    NUM_VARS = 5

    (
        adjacency_mat,
        weighted_mat,
        exogenous_nodes,
        endogenous_nodes,
        outcome_nodes,
        scm,
    ) = _random_causal_graph(
        nb_nodes=NUM_VARS,
        p=0.3,
        low=0.0,
        high=5.0,
        binomial_exogenous_variables=False,
        binomial_probability=0.6,
    )
    SIZE = 10_000
    X_df_2 = scm.sample(SIZE)

    scm.cgm.draw().view()

    (adj_mat_2, arrows_mat_2), probes_2 = ic_star(X_df_2)
    print(f"Variables: {X_df_2.columns.sort_values()}")
    print(f"Adjacency matrix:\n{adj_mat_2}")
    print(f"Arrows matrix:\n{arrows_mat_2}")
    ic_star_graph_2 = nx.from_numpy_array(adj_mat_2)
    pos = graphviz_layout(ic_star_graph_2, prog="dot")
    fig = plt.figure()
    plt.title("IC Star Graph")
    nx.draw(ic_star_graph_2, pos)
    labels = { i : f"x{i}" for i in range(len(adj_mat_2))}
    nx.draw_networkx_labels(ic_star_graph_2, pos, labels)

    # something is buggy right here
    variable_types_2 = {name: "c" for name in X_df_2.columns}
    graph_comp_2 = IC(RobustRegressionTest).search(X_df_2, variable_types_2)
    for i, j in graph_comp_2.edges():
        print(i, j, graph_comp_2.get_edge_data(i, j))

    pos = graphviz_layout(graph_comp_2, prog="dot")
    fig = plt.figure()
    plt.title("IC Graph")
    nx.draw(graph_comp_2, pos, with_labels=True)
    
    plt.show()

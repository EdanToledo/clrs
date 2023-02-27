import time
from matplotlib import pyplot as plt
import networkx as nx

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest


from clrs._src.causal_data_generation import _random_causal_graph

NUM_NODES = 5

(
    adjacency_mat,
    weighted_mat,
    exogenous_nodes,
    endogenous_nodes,
    outcome_nodes,
    scm,
) = _random_causal_graph(
    nb_nodes=NUM_NODES,
    p=0.3,
    low=0.0,
    high=5.0,
    binomial_exogenous_variables=False,
    binomial_probability=0.6,
)

scm.cgm.draw().view()

# generate some toy data:
SIZE = 10_000
X = scm.sample(SIZE)

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.
variable_types = {f"x{i}" : "c" for i in range(NUM_NODES)}

# run the search
ic_algorithm = IC(RobustRegressionTest)
start = time.time()
graph = ic_algorithm.search(X, variable_types)
end = time.time()
print(end-start)
edge_data = graph.edges(data=True)
print(edge_data)
for edge in edge_data:
    direction = edge[2]["arrows"]
    if direction:
        if edge[0] == direction[0]:
            print(edge[1], "-->", direction[0])
        else:
            print(edge[0], "-->", direction[0])


# Visualise Identified Structure
pos = nx.planar_layout(graph)
nx.draw(graph, pos, with_labels=True)
plt.show()

import numpy as np
import networkx as nx
import string
from itertools import chain, product, islice, count

from numpy.random import default_rng
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling



#def sample_prototype(nodes=)
def sample_key_door_goal(nodes=['Key', 'Door', 'Goal'], num_variables=3,
                            rng=default_rng()):
    "K->D, D->G"
    # model = BayesianNetwork([('Key', 'Door'), ('Door', 'Goal')])
    adjacency = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
                ])

    num_variables = len(adjacency)

    if nodes is None:
            uppercase = string.ascii_uppercase
            iterator = chain.from_iterable(
                product(uppercase, repeat=r) for r in count(1))
            nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph

def sample_coin_flip(nodes=None, num_variables=4, rng=default_rng()):
    """A->B, A->C, B->D, C->D"""
       
    # Create a Bayesian Network
    model = BayesianNetwork([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])

    # Define Conditional Probability Distributions (CPDs)
    cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]])

    cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.7, 0.3], [0.3, 0.7]],
                    evidence=['A'], evidence_card=[2])

    cpd_C = TabularCPD(variable='C', variable_card=2, values=[[0.6, 0.4], [0.4, 0.6]],
                    evidence=['A'], evidence_card=[2])

    cpd_D = TabularCPD(variable='D', variable_card=2, values=[[0.8, 0.4, 0.7, 0.2],
                                                            [0.2, 0.6, 0.3, 0.8]],
                    evidence=['B', 'C'], evidence_card=[2, 2])
    
    model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)   # Add CPDs to the model
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    return model

def sample_rain_grass_wet(nodes=None, num_variables=3, rng=default_rng()):
    """R->S, R->G, S->G"""
       
    # Create a Bayesian Network
    model = BayesianNetwork([('R', 'S'), ('R', 'G'), ('S', 'G')])

    # Define Conditional Probability Distributions (CPDs)
    cpd_R = TabularCPD(variable='R', variable_card=2, values=[[0.8], [0.2]])

    cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.4, 0.01], [0.6, 0.99]],
                    evidence=['R'], evidence_card=[2])

    cpd_G = TabularCPD(variable='G', variable_card=2, values=[[0.0, 0.8, 0.9, 0.99],
                                                            [1.0, 0.2, 0.1, 0.01]],
                    evidence=['R', 'S'], evidence_card=[2, 2])
    
    model.add_cpds(cpd_R, cpd_S, cpd_G)   # Add CPDs to the model
    # graph = nx.DiGraph()                         # Create a directed graph from the model
    assert model.check_model()              # Check the model for consistency
    model.add_nodes_from(model.nodes())     # Add nodes (variables) to the graph
    model.add_edges_from(model.edges())     # Add edges (dependencies) to the graph
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    return model

def sample_grid_world_2x2(nodes=None,
                            rng=default_rng()
    ): 
    # adjacency = np.array([
    #             [0, 1, 1, 0],
    #             [0, 0, 0, 1],
    #             [0, 0, 0, 1],
    #             [0, 0, 0, 0],
    #             ])
    adjacency = np.array([
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                ])

    num_variables = len(adjacency)

    if nodes is None:
            uppercase = string.ascii_uppercase
            iterator = chain.from_iterable(
                product(uppercase, repeat=r) for r in count(1))
            nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    # pos = nx.spring_layout(graph, k=0.15, iterations=80)
    # nx.draw(graph, pos, with_labels = True)

    return graph

def sample_grid_world_3x3(nodes=None,
                            rng=default_rng()
    ): 
    adjacency = np.array([
                [0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
    # adjacency = np.array([
    #             [0, 1, 0, 1, 0, 0, 0, 0, 0],
    #             [1, 0, 1, 0, 1, 0, 0, 0, 0],
    #             [0, 1, 0, 0, 0, 1, 0, 0, 0],
    #             [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #             [0, 0, 0, 0, 0, 1, 0, 1, 0],
    #             [0, 0, 1, 0, 1, 0, 0, 0, 1],
    #             [0, 0, 0, 1, 0, 0, 0, 1, 0],
    #             [0, 0, 0, 0, 1, 0, 1, 0, 1],
    #             [0, 0, 0, 0, 0, 1, 0, 1, 0],
    #             ])


    num_variables = len(adjacency)

    if nodes is None:
            uppercase = string.ascii_uppercase
            iterator = chain.from_iterable(
                product(uppercase, repeat=r) for r in count(1))
            nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph

def sample_grid_world_4x4(nodes=None,
                            rng=default_rng()
    ): 
    adjacency = np.array([
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
             ])
    num_variables = len(adjacency)

    if nodes is None:
            uppercase = string.ascii_uppercase
            iterator = chain.from_iterable(
                product(uppercase, repeat=r) for r in count(1))
            nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    # pos = nx.spring_layout(graph, k=0.15, iterations=80)
    # nx.draw(graph, pos, with_labels = True)

    return graph

def sample_erdos_renyi_graph(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        create_using=models.BayesianNetwork,
        rng=default_rng()
    ):
    if p is None:
        if num_edges is None:
            raise ValueError('One of p or num_edges must be specified.')
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.)
    
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph

def sample_erdos_renyi_linear_gaussian(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=default_rng()
    ):
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.LinearGaussianBayesianNetwork,
        rng=rng
    )

    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.  # There is no bias term

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)
    return graph


def sample_erdos_renyi_dirichlet_multinomial(
        num_variables,
        cardinalities,
        p=None,
        num_edges=None,
        nodes=None,
        alpha=1.,
        rng=default_rng()
    ):
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.BayesianNetwork,
        rng=rng
    )
    nodelist = list(graph.nodes)

    if isinstance(cardinalities, int):
        cardinalities = [cardinalities for _ in range(num_variables)]
    
    if len(cardinalities) != num_variables:
        raise ValueError(f'The length of cardinalities ({cardinalities}) must '
            f'be equal to the number of variables ({num_variables}).')

    # Create the model parameters
    factors = []
    for idx, node in enumerate(nodelist):
        parents = list(graph.predecessors(node))
        parent_indices = [
            index for (index, parent) in enumerate(nodelist)
            if parent in parents
        ]
        parent_cards = [cardinalities[index] for index in parent_indices]
        num_columns = max(1, int(np.prod(parent_cards)))

        # Sample random parameters (from Dirichlet distribution)
        alpha_vector = np.full((cardinalities[idx],), alpha)
        thetas = rng.dirichlet(alpha_vector, size=(num_columns,))

        # Create factor
        factor = TabularCPD(
            node,
            cardinalities[idx],
            thetas.T,
            parents,
            parent_cards    
        )
        factor.normalize()
        factors.append(factor)

    graph.add_cpds(*factors)
    return graph


def _s(node1, node2):
    return (node2, node1) if (node1 > node2) else (node1, node2)

def get_markov_blanket(graph, node):
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))

    mb_nodes = parents | children
    for child in children:
        mb_nodes |= set(graph.predecessors(child))
    mb_nodes.discard(node)

    return mb_nodes


def get_markov_blanket_graph(graph):
    """Build an undirected graph where two nodes are connected if
    one node is in the Markov blanket of another.
    """
    # Make it a directed graph to control the order of nodes in each
    # edges, to avoid mapping the same edge to 2 entries in mapping.
    mb_graph = nx.DiGraph()
    mb_graph.add_nodes_from(graph.nodes)

    edges = set()
    for node in graph.nodes:
        edges |= set(_s(node, mb_node)
            for mb_node in get_markov_blanket(graph, node))
    mb_graph.add_edges_from(edges)

    return mb_graph


def adjacencies_to_networkx(adjacencies, nodes):
    mapping = dict(enumerate(nodes))
    for adjacency in adjacencies:
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        yield nx.relabel_nodes(graph, mapping, copy=False)


if __name__ == '__main__':
    from dag_gflownet.utils.sampling import sample_from_discrete

    nodes = ['diff', 'intel', 'grade']
    cardinalities = [2, 3, 3]
    rng = default_rng(0)

    graph = sample_erdos_renyi_dirichlet_multinomial(
        3, cardinalities, num_edges=2, nodes=nodes, rng=rng)

    data = sample_from_discrete(graph, 100, rng=rng)

    print(data.head(10))

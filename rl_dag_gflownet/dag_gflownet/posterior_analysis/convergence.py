import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import matplotlib.colors as mcolors
import math


def avg_presence_edges(posterior):
    """
    Compute the average presence of each edge in the posterior data.
    Edges with values close to 1 are almost always present in the samples.
    Edges with values close to 0 are rarely present.
    Edges with values around 0.5 have an uncertain or variable presence.
    Args:
        posterior (np.ndarray): Posterior data with shape (num_samples, num_edges).

    Returns:
        np.ndarray: Average presence of each edge in the posterior data.
    """
    edge_presence_avg = np.mean(posterior, axis=0)

    print("Average presence of each edge:")
    print(edge_presence_avg)
    num_edges = edge_presence_avg.shape[0]

    # Create a bar graph
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(edge_presence_avg)), edge_presence_avg, color='skyblue')  # Correct usage of bar()

    # plt.xlabel('Edge Index')
    # plt.ylabel('Average Presence')
    # plt.title('Average Presence of Each Edge in Posterior')
    # plt.savefig('avg_presence_edges_kd3_10k.png')

    return edge_presence_avg


def vis_graph_avg_presence(edge_avg_presence, data):
   
    # Load the posterior data (Replace with your actual data and file path)
    # Example graph (Replace with your actual graph)
    
    visited_edges = []
    for adjacency_matrix in data:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if adjacency_matrix[i][j] == 1:
                    visited_edges.append((i,j))

    edge_visitation_counts = {}
    for visit in visited_edges:
        edge=tuple(visit)
        if edge in edge_visitation_counts:
            edge_visitation_counts[edge] +=1
        else:
            edge_visitation_counts[edge] = 1

    # Extract visitation counts from the dictionary
    edge_colors = [visit_count for edge, visit_count in edge_visitation_counts.items()]

    # Normalize edge colors to fall within a specific range (e.g., 0 to 1)
    min_value = min(edge_colors)
    max_value = max(edge_colors)
    normalized_colors = [(visit_count - min_value) / (max_value - min_value) for visit_count in edge_colors]

    # Create a color map to map normalized values to colors (e.g., from blue to red)
    cmap = plt.get_cmap('viridis')

    # Create a directed graph with custom edge attributes
    DG = nx.DiGraph(adjacency_matrix)
    # pos = nx.spring_layout(DG, seed=43)

    # Add nodes
    # DG.add_nodes_from(data[0].nodes())
    print(DG.nodes())
    node_color = 'skyblue'
    #node_labels = ['K', 'D', 'G']
    #node_labels = {0: 'Di0', 1: 'Di1', 2: 'K1', 3: 'k3p', 4: 'k3', 5:'D', 6: 'DD', 7:'K2'}
    node_labels = {0: 'D', 1: 'K1', 2: 'K3', 3: 'D', 4: 'k2'}

    #node_labels = {node: node_labels[node] for node in DG.nodes()}
    #nodes = [(i * cols + j) for i in range(rows) for j in range(cols)]

    for variable in DG.nodes():
        DG.add_node(variable, label=variable)

    rows, cols = 4, 4
    pos = {}
    for node in DG.nodes:
        j = node % rows
        i = math.floor(node / rows)
        pos[node] = (j , -i)

    # pos = nx.spring_layout(DG, scale=0.25)#nx.spring_layout(G, k=0.2, iterations=80)

    nx.draw_networkx_nodes(DG, pos, node_color=node_color, node_size=300)
    nx.draw_networkx_labels(DG, pos, labels=node_labels, font_size=12, font_color='black', font_weight='bold')

    # Add edges with direction and color attributes
    for edge, visit_count in edge_visitation_counts.items():
        source, target = edge
        edge_color = cmap((visit_count - min_value) / (max_value - min_value))
        DG.add_edge(source, target, color=edge_color, visit_count=visit_count)

    # Plot the DAG with different arrow directions and colors
    edge_labels_ = {(source, target): f'{data["visit_count"]}' for source, target, data in DG.edges(data=True)}
    sorted_edge_labels = {k: v for k, v in sorted(edge_labels_.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_edge_labels)

    # Extract edge colors from the edge attributes
    edge_colors = [data['color'] for source, target, data in DG.edges(data=True)]
    print(edge_colors)

    # Draw edges with arrows and different colors
    nx.draw_networkx_edges(DG, pos, edgelist=DG.edges(), edge_color=edge_colors, width=2, arrowsize=20, connectionstyle="arc3,rad=0.6")

    # Create a color bar for edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm.set_array([])  # To make sure the color bar covers the full range
    cbar = plt.colorbar(sm, label='Edge Visitation Frequency')

    # Set plot title
    plt.title("Full graph: edge visitation")

    # Show the plot
    plt.axis('off')
    plt.savefig('graph_avg_presence.png')




if __name__ == "__main__":

    current_working_directory = os.getcwd()
    print(f"Current Working Directory: {current_working_directory}")
    # Load the posterior data
    posterior = np.load('posterior_analysis/posterior_files/kd3-10k-posterior-2000-5p.npy')  # Replace with your file path
    print(posterior.shape)
    edge_avg_presence = avg_presence_edges(posterior)
    #vis_graph_avg_presence(edge_avg_presence, posterior)
    # Assuming the first dimension represents different samples
    



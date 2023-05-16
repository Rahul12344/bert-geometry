from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import heapq

def compute_pairwise_distances(tokens, dependency_edges, max_sentence_length):
    # convert dependency edges to graph
    graph = {}
    for u, v,_ in dependency_edges:
        if u not in graph:
            graph[u] = {}
        graph[u][v] = 1
        
    for u, v, _ in dependency_edges:
        if v not in graph:
            graph[v] = {}
        graph[v][u] = 1
        
    pairwise = np.zeros((max_sentence_length, max_sentence_length))
    
    for i in range(len(tokens)):
        distances = dijkstra(tokens, graph, i)
        for node in distances:
            pairwise[i][node] = distances[node]
    return pairwise

def dijkstra(tokens, graph, start):
    distances = {node: float('inf') for node in range(0,len(tokens))}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node in graph:
            for neighbor, weight in graph[current_node].items():
                distance = current_dist + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

    return distances

def calculate_mst(pairwise_distances):
    sparse_pw_distances = csr_matrix(pairwise_distances)
    mst = minimum_spanning_tree(sparse_pw_distances)
    mst = mst.toarray().astype(float)
    return mst

def plot_mst(mst, tokens, save_file_name):
    G = nx.Graph(mst)
    mapping = {i: token for i, token in enumerate(tokens) if i in G.nodes()}
    G = nx.relabel_nodes(G, mapping, copy=False)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color="grey")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): round(d["weight"], 1) for u, v, d in G.edges(data=True)}
    )
    nx.draw_networkx_edges(G, pos, edge_color="green", width=2)
    plt.axis("off")
    plt.savefig(save_file_name)
    
def plot_mst_from_dependency_edges(dependency_edges, words, save_file_name):
    # remove third element from each tuple
    dependency_edges_ = [(u, v, 1.0) for u, v, _ in dependency_edges]
    
    # create mapping from index to word
    mapping = {i: word for i, word in enumerate(words)}
    
    G = nx.Graph()
    G.add_weighted_edges_from(dependency_edges_)
    
    # relabel nodes
    G = nx.relabel_nodes(G, mapping, copy=False)
    
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color="grey")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): round(d["weight"], 1) for u, v, d in G.edges(data=True)}
    )
    nx.draw_networkx_edges(G, pos, edge_color="green", width=2)
    plt.axis("off")
    plt.savefig(f'{save_file_name}.png')
    plt.clf()
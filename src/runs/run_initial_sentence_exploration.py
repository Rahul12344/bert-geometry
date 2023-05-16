import os, sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, Isomap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from transformers import BertTokenizer, BertModel

# get parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

from config.config import config
from embedder.create_sentence_embeddings import generate_token_embeddings
from utils.build_mst_from_pairwise import plot_mst, plot_mst_from_dependency_edges, compute_pairwise_distances
from utils.build_individual_parse_tree import get_dependency_trees

if __name__ == '__main__':
    c = config()
    
    # initialize model and tokenizer
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # create sample sentence with dependency parse tree
    sample_sentence = "we are trying to understand the difference"
    words = sample_sentence.split()
    dep_structures = {(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,6,1),(5,6,1)}

    plot_mst_from_dependency_edges(dep_structures, words)
    
    tokens, sample_embeddings = generate_token_embeddings(sample_sentence, model, tokenizer)
    sample_embeddings = sample_embeddings.cpu().detach().numpy()
    
    # plot dendogram of embeddings to see hierarchy of clusters
    Z = linkage(sample_embeddings, method='complete')
    dendrogram(Z, labels=tokens)
    plt.savefig('sample_dendrogram.png')
    plt.clf()

    # plot heatmap of dimensionality-reduced embeddings to see hierarchy of clusters
    iso = Isomap(n_components=3)
    sample_reduction = iso.fit_transform(sample_embeddings)
    
    distances = euclidean_distances(sample_embeddings)
    reduced_distances = euclidean_distances(sample_reduction)
    similarities = cosine_similarity(sample_reduction)
    
    
    sns.heatmap(reduced_distances, square=True, annot=True, cbar=False, cmap='Blues')
    plt.savefig('sample_distances.png')
    plt.clf()
    sns.heatmap(similarities, square=True, annot=True, cbar=False, cmap='Blues')
    plt.savefig('sample_similarities.png')
    plt.clf()
    
    # plot mst of embeddings to see hierarchy of clusters
    reduced_distances = csr_matrix(reduced_distances)
    Tcsr = minimum_spanning_tree(reduced_distances)
    plot_mst(Tcsr.toarray(), tokens, 'sample_reduced_mst_tok.png')
    
    # plot mst of embeddings to see hierarchy of clusters
    distances = csr_matrix(distances)
    Tcsr = minimum_spanning_tree(distances)
    plot_mst(Tcsr.toarray(), tokens, 'sample_mst_tok.png')
    

    
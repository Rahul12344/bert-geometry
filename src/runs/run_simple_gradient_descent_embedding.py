import os, sys
import seaborn as sns
import matplotlib.pyplot as plt

# get parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

from config.config import config
from utils.build_mst_from_pairwise import compute_pairwise_distances, plot_mst, calculate_mst
from utils.gradient_descent import gradient_descent_individual, initialize_locations_individual

if __name__ == '__main__':
    c = config()
    
    # create sample sentence with dependency parse tree
    sample_sentence = "we are trying to understand the difference"
    words = sample_sentence.split()
    dep_structures = {(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,6,1),(5,6,1)}
    
    discrepancy_matrix = compute_pairwise_distances(words, dep_structures)
    init_loc = initialize_locations_individual(len(words))
    
    # gradient descent for single sentence embedding
    optimal_embeddings, dists = gradient_descent_individual(init_loc, discrepancy_matrix, len(words))
    mst = calculate_mst(dists)
    plot_mst(mst, words, 'mst_from_gradient_descent.png')
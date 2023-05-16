import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def compute_discrepancy_individual(locations, dist_matrix, number_locations):
    # Compute the distances between the cities in R2
    dists = compute_pairwise_distances_individual(locations)
    # Compute the difference between the distances in the distance matrix and the distances in R2
    diff = dists - dist_matrix
    # Compute the cost function
    cost = 0
    for i in range(number_locations):
        for j in range(number_locations):
            if i != j:
                cost += diff[i, j]**2
    return cost

def initialize_locations_individual(number_locations):
    locations = np.random.rand(number_locations, 1024)
    return locations

def compute_pairwise_distances_individual(locations):
    dists = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=-1)
    return dists

# Compute the gradient of the cost function
def gradient_individual(locations, dist_matrix, number_locations):
    # Compute the pairwise distances between the cities using some distance metric
    dists = compute_pairwise_distances_individual(locations)
    
    # Compute the difference between the distances in the distance matrix and the distances in some metric space
    diff = dists - dist_matrix

    # initialize the gradient
    grad = np.zeros_like(locations)
    for i in range(number_locations):
        for j in range(number_locations):
            if i != j:
                # multiplying constant doesn't matter
                grad[i] += 4 * diff[i, j] * 1/dists[i,j] * (locations[i] - locations[j])
    return grad

def gradient_descent_individual(locations, dist_matrix, number_locations, learning_rate=0.0001, number_iterations=10000, tolerance=1e-3):
    for i in range(number_iterations):
        start_distance = compute_discrepancy_individual(locations, dist_matrix, number_locations)
        # Compute the gradient of the discrepancy function
        grad = gradient_individual(locations, dist_matrix, number_locations)
        # Update the locations of the cities using gradient descent
        locations -= learning_rate * grad
        
        end_distance = compute_discrepancy_individual(locations, dist_matrix, number_locations)
        if abs(start_distance - end_distance) < tolerance:
            print('Converged after {} iterations'.format(i))
            break
     
    # Compute the pairwise distances between the cities using some distance metric
    print('Computing pairwise distances...')
    end_distance = compute_discrepancy_individual(locations, dist_matrix, number_locations)
    print(end_distance)
    dists = np.linalg.norm(locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=-1)
    return locations, dists

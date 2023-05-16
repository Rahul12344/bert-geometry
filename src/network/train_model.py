import torch
import os, sys
from torch.optim import Adam
from tqdm import tqdm
import logging
logging.basicConfig(level = logging.INFO)
import os

# get parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

# class to find optimal weights for a given set of inputs and outputs
class FindOptimalWeights:
    
    def __init__(self, c, regressor, epochs=10000, learning_rate=0.001):
        self.c = c
        self.regressor = regressor
        self.epochs = epochs
        self.optimizer = Adam(regressor.parameters(), lr=learning_rate)
    
    def stochastic_gradient_descent(self, cost_function, train_dataset):
        for epoch in tqdm(range(self.epochs)):
            running_train_loss = 0.0
            num_training_batches = 0 
            for batch in tqdm(train_dataset):
                num_training_batches += 1
                embeddings, distances = batch
                
                self.optimizer.zero_grad()
                predicted_distances = self.regressor(embeddings)
                
                loss = cost_function(predicted_distances, distances)
                loss.backward()
                running_train_loss += loss.item()
                self.optimizer.step()
                
            logging.info("Epoch {} - Training Loss: {}". format(epoch+1, running_train_loss/num_training_batches))
    
    def predict(self, embeddings):
        return self.regressor(embeddings)



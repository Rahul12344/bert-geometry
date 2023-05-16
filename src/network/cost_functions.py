import torch
import torch.nn as nn
import math

# Define cost functions to compute the distance between embedding
# and optimal alignment, where we transform the embedding using
# an isometry

# Compute the cost function, we will compute the gradient of this
# to update the weights and perform gradient descent
class MSECostFunction(nn.Module):
    
    # initialize the cost function
    def __init__(self, device):
        super(MSECostFunction, self).__init__()
        self.device = device

    # compute the cost function
    def forward(self, output, target):
        num_sentences, max_sentence_length, _ = output.size()
        loss = torch.tensor(0.0).to(self.device)
        for sentence_num in range(num_sentences):
            # get the output for the sentence
            sentence_output = output[sentence_num]
            # get the target for the sentence
            sentence_target = target[sentence_num]
            # compute the normalized loss
            loss += torch.mean((sentence_output - sentence_target) ** 2)
        loss /= num_sentences
        return loss
        
class L1CostFunction(nn.Module):
    
    # initialize the cost function
    def __init__(self, device):
        super(L1CostFunction, self).__init__()
        self.device = device

    # compute the cost function
    def forward(self, output, target):
        num_sentences, max_sentence_length, _ = output.size()
        loss = torch.tensor(0.0).to(self.device)
        for sentence_num in range(num_sentences):
            # get the output for the sentence
            sentence_output = output[sentence_num]
            # get the target for the sentence
            sentence_target = target[sentence_num]
            # compute the normalized loss
            loss += torch.mean(torch.abs(sentence_output - sentence_target))
        loss /= num_sentences
        return loss
    
class MSEWSDCostFunction(nn.Module):
    
    # initialize the cost function
    def __init__(self, device):
        super(MSEWSDCostFunction, self).__init__()
        self.device = device

    # compute the cost function
    def forward(self, output, target):
        loss = torch.tensor(0.0).to(self.device)
        # compute the normalized loss
        loss += torch.mean((output - target) ** 2)
        return loss
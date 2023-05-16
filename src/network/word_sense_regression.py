import torch
import torch.nn as nn
import math

# Define regression model where we transform the embedding using
# an isometry

# Compute the forward function of the linear regression model
class WordSenseOptimizer(nn.Module):
    # initialize the optimizer
    def __init__(self, **kwargs):
        super(WordSenseOptimizer, self).__init__()
        self.c = kwargs["c"]
        self.device = kwargs["device"]
        self.weights = nn.parameter.Parameter(torch.zeros(self.c['layer_size'], self.c['matrix_rank']))
        nn.init.uniform_(self.weights, -0.05, 0.05)
        self.to(self.device)
        
    # compute the forward function
    # input: sentences, a list of sentences (_, sentence_length, embedding_size)
    # output: predicted pairwise distances
    # we pad the sentences to the max sentence length
    def forward(self, sentences):
        # apply the linear transformation to the embeddings
        if self.c['log_normalize']:
            sentences = torch.log(torch.abs(sentences) + 1)
        if self.c['sqrt_normalize']:
            sentences = torch.sqrt(torch.abs(sentences))
            
        projected_embeddings = torch.matmul(sentences, self.weights)
        
        if self.c['logits']:
            projected_embeddings = torch.sigmoid(projected_embeddings)
            
        if self.c['elu']:
            projected_embeddings = nn.functional.elu(projected_embeddings)
        
        # compute the pairwise distances within words
        word_distance = torch.cdist(projected_embeddings, projected_embeddings)
            
        
        return word_distance.to(self.device)
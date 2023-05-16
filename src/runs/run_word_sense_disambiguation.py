import os, sys
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get parent directory of this codebase
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

from config.config import config
from utils.lexsub_xml import read_lexsub_xml, get_position_of_target
from embedder.create_sentence_embeddings import generate_token_embeddings, match_tokenized_to_untokenized, get_word_embeddings
from datasets.dataset import DistanceDataset, get_dataset_loaders
from network.word_sense_regression import WordSenseOptimizer
from network.cost_functions import MSEWSDCostFunction
from network.train_model import FindOptimalWeights

if __name__ == '__main__':
    c = config()
    
    sentences = []
    word_positions = []
    pos = []
    
    number_samples = 0
    for context in tqdm(read_lexsub_xml(os.path.join(c['word_senses_dir'], 'lexsub.xml'))):
        if number_samples == c['max_data_samples']:
            break
        sentence = context.left_context  + [context.lemma] + context.right_context 
        sentences.append(sentence)
        word_position = get_position_of_target(context)
        word_positions.append(word_position)
        pos.append(context.pos)
        number_samples += 1
        
    # initialize BERT model
    # initialize model and tokenizer
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # compute embeddings for each sentence
    sentences_embeddings = []
    for i, sentence in tqdm(enumerate(sentences)):
        # get BERT embeddings for each token in the sentence
        tokens, embeddings = generate_token_embeddings(' '.join(sentence), model, tokenizer)
        
        # map subword tokens to untokenized sentence
        mapper = match_tokenized_to_untokenized(tokens, sentence)
        
        # average subword embeddings to get word embeddings
        embeddings = get_word_embeddings(embeddings, mapper)
        sentences_embeddings.append(embeddings)
    
    # index of target word in sentence
    word_embedding = []
    for i, position in enumerate(word_positions):
        word_embedding.append(sentences_embeddings[i][position])
    word_embedding = torch.stack(word_embedding)
    
    Z = linkage(word_embedding.detach().cpu().numpy(), metric='cosine', method='complete')
    dendrogram(Z, labels=pos)
    plt.savefig('wsd_test.png')
    plt.clf()
    
    df = pd.read_csv(os.path.join(c['word_senses_dir'], 'disc.csv'))
    discrepancy = df.to_numpy()
    
    sns.heatmap(discrepancy, square=True, annot=True, cbar=False, cmap='Purples')
    plt.savefig('brighttrue.png')
    plt.clf()
    
    # create dataset
    discrepancy = torch.from_numpy(discrepancy).float()
    distance_dataset = DistanceDataset(c, device, word_embedding, discrepancy)
    
    # initialize regressor
    regressor = WordSenseOptimizer(c=c, device=device)
    
    # initialize cost function
    cost = MSEWSDCostFunction(device)
    
    # initialize trainer
    trainer = FindOptimalWeights(c, regressor)
    train_dataset = get_dataset_loaders(distance_dataset, c['batch_size'])
    trainer.stochastic_gradient_descent(cost, train_dataset)
    
    predicted_distances = regressor(word_embedding.to(device))
    predicted_distances = predicted_distances.cpu().detach().numpy()
    predicted_distances = predicted_distances > 0.6
    sns.heatmap(predicted_distances, square=True, annot=True, cbar=False, cmap='Purples')
    plt.savefig('brightpred_rounded.png')
    plt.clf()
    
    
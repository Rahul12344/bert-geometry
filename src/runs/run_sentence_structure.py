import sys, os
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import ClusterWarning
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)


# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

from config.config import config
from utils.build_individual_parse_tree import get_dependency_trees
from utils.build_mst_from_pairwise import compute_pairwise_distances
from embedder.create_sentence_embeddings import generate_token_embeddings, generate_sentence_embeddings, match_tokenized_to_untokenized, get_word_embeddings
from datasets.dataset import DistanceDataset, get_dataset_loaders
from network.linear_regression import TransformOptimizer
from network.cost_functions import MSECostFunction, L1CostFunction
from network.train_model import FindOptimalWeights
from utils.build_mst_from_pairwise import calculate_mst, plot_mst, plot_mst_from_dependency_edges
from utils.gradient_descent import gradient_descent_individual, initialize_locations_individual
from clustering_dist.hausdorff import estimate, diameter

if __name__ == '__main__':
    c = config()
    
    # initialize BERT model
    # initialize model and tokenizer
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model_sent = SentenceTransformer('all-MiniLM-L6-v2')
    
    # get dataset of parse trees (limited dataset due to system and time constraints)
    all_sentences, parse_structures, sentence_lengths, max_sentence_length = get_dependency_trees(os.path.join(c['sentence_dir'], 'train.conll'), start_idx=c['start_data_samples'], max_data_samples=c['max_data_samples'])

    pw_distances = [compute_pairwise_distances(sentence, parse_structure, max_sentence_length) for sentence, parse_structure in tqdm(zip(all_sentences, parse_structures))]
    pw_distances = np.array(pw_distances)
    
    # convert discrepancy matrix to tensor
    pw_distances = torch.from_numpy(pw_distances)
        
    # compute embeddings for each sentence
    sentences_embeddings = []
    for sentence in tqdm(all_sentences):
        # get BERT embeddings for each token in the sentence
        tokens, embeddings = generate_token_embeddings(' '.join(sentence), model, tokenizer)
        # map subword tokens to untokenized sentence
        mapper = match_tokenized_to_untokenized(tokens, sentence)
        
        # average subword embeddings to get word embeddings
        embeddings = get_word_embeddings(embeddings, mapper)
        sentences_embeddings.append(embeddings)
        
    # pad embeddings to max sentence length
    sentences_embeddings = pad_sequence(sentences_embeddings, batch_first=True, padding_value=0)
    
    if c['scale']:
        m = sentences_embeddings.mean(0, keepdim=True)
        s = sentences_embeddings.std(0, unbiased=False, keepdim=True)
        sentences_embeddings -= m
        sentences_embeddings /= s
    
    
    # create dataset
    distance_dataset = DistanceDataset(c, device, sentences_embeddings, pw_distances)
    
    # initialize regressor
    regressor = TransformOptimizer(c=c, device=device)
    
    # initialize cost function
    cost = MSECostFunction(device)
    
    # initialize trainer
    trainer = FindOptimalWeights(c, regressor)
    train_dataset = get_dataset_loaders(distance_dataset, c['batch_size'])
    trainer.stochastic_gradient_descent(cost, train_dataset)
    
    predicted_distances = regressor(sentences_embeddings.to(device))
    predicted_distances = predicted_distances.cpu().detach().numpy()
    pw_distances = pw_distances.cpu().detach().numpy()
    
    # plot MSTs and basic heatmaps
    predicted_distances_0 = predicted_distances[0]
    mst = calculate_mst(predicted_distances_0[:sentence_lengths[0],:sentence_lengths[0]])
    plot_mst_from_dependency_edges(parse_structures[0], all_sentences[0], 'test0true')
    plt.clf()
    plot_mst(mst, all_sentences[0], 'test0pred')
    plt.clf()
    
    sns.heatmap(np.round(pw_distances[0][:sentence_lengths[0],:sentence_lengths[0]],1), square=True, annot=True, cbar=False, cmap='Blues')
    plt.savefig('test0trueheatmap.png')
    plt.clf()
    sns.heatmap(np.round(predicted_distances_0[:sentence_lengths[0],:sentence_lengths[0]],1), square=True, annot=True, cbar=False, cmap='Blues')
    plt.savefig('test0predheatmap.png')
    plt.clf()
    
    init_locations = initialize_locations_individual(sentence_lengths[0])
    embedded_locations, dists = gradient_descent_individual(init_locations, pw_distances[0][:sentence_lengths[0],:sentence_lengths[0]], sentence_lengths[0])
    mst = calculate_mst(dists)
    plot_mst(mst, all_sentences[0], 'test0desc.png')
    plt.clf()
    sns.heatmap(np.round(dists,1), square=True, annot=True, cbar=False, cmap='Blues')
    plt.savefig('test0descheatmap.png')
    plt.clf()
    
    sentences_embeddings = []
    sentences = [['the', 'dog', 'ran'], ['the', 'cat', 'ran'], ['he', 'is', 'happy'], ['she', 'is', 'happy'], ['dog', 'is', 'very', 'happy']]
    for sentence in tqdm(sentences):
        tokens, embeddings = generate_token_embeddings(' '.join(sentence), model, tokenizer)
        mapper = match_tokenized_to_untokenized(tokens, sentence)
        
        # average subword embeddings to get word embeddings
        embeddings = get_word_embeddings(embeddings, mapper)
        sentences_embeddings.append(embeddings)
        
    # pad embeddings to max sentence length
    sentences_embeddings = pad_sequence(sentences_embeddings, batch_first=True, padding_value=0)
        
    gamma = regressor.weights
    diameters = []
    transformed_sentences_embeddings = torch.matmul(sentences_embeddings.to(device), gamma.to(device))
    for transformed_sentence in transformed_sentences_embeddings:
        transformed_sentence = transformed_sentence.cpu().detach().numpy()
        print(transformed_sentence)
        diameters.append(diameter(transformed_sentence))
    
    D = estimate(diameters)
    print(D)
    Z = linkage(D, method='complete')
    dendrogram(Z)
    plt.savefig("grhscluster_cat.png")
    plt.clf()
    
    sentences = ['the dog ran', 'the cat ran', 'he is happy', 'she is happy', 'dog is very happy']
    embeddings = generate_sentence_embeddings(sentences, model_sent)
    print(embeddings.shape)
    distances = 1 - cosine_similarity(embeddings)
    Z = linkage(distances, method='average', metric='cosine')
    dendrogram(Z)
    plt.savefig("sentclust.png")
    plt.clf()
    
    
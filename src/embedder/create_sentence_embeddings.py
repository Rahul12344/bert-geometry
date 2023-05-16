from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict

# modified from Hewitt and Manning 2019
def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    token_mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 0
    while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[tokenized_sent_index + 1].startswith('##')):
            token_mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        token_mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return token_mapping

def get_word_embeddings(sentence_embedding, mapping):
    word_embeddings = []
    map_keys = sorted(list(mapping.keys()))
    for key in map_keys:
        if len(mapping[key]) == 1:
            word_embeddings.append(sentence_embedding[mapping[key][0]])
        else:
            word_embeddings.append(torch.mean(sentence_embedding[mapping[key][0]:mapping[key][-1]], dim=0))
    return torch.stack(word_embeddings)

def generate_token_embeddings(sentence, model, tokenizer):
    # Tokenize input sentence
    tokens = tokenizer.tokenize(sentence)
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    # Generate token embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        token_embeddings = outputs.last_hidden_state

    return tokens, token_embeddings.squeeze(0)

def generate_sentence_embeddings(sentences, model):
    return model.encode(sentences)
import os, sys
from tqdm import tqdm

# get parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

from utils.extract_training_data import get_training_instances
from utils.conll_reader import conll_reader

def get_dependency_trees(data_files, start_idx=0, max_data_samples=1000):
    """
    Returns a list of DependencyStructure objects extracted from the data
    in data_files
    """
    sentence_lengths = []
    words = []
    parse_structures = []
    current_num_samples = 0
    max_sentence_length = 0
    idx = 0
    with open(data_files, 'r') as data_file:
        for dtree in tqdm(conll_reader(data_file)):
            if idx >= start_idx:
                sentence = dtree.words()
                sentence = [word.lower() for word in sentence if word != None]
                words.append(sentence)
                sentence_lengths.append(len(sentence))
                max_sentence_length = max(max_sentence_length, len(sentence))
                
                parse_structure = get_training_instances(dtree)
                parse_structure = parse_structure[-1][-2].deps
                filtered_parse_structure = []
                for structure in parse_structure:
                    if structure[0] != 0:
                        filtered_parse_structure.append((structure[0]-1, structure[1]-1, 1))
                parse_structures.append(filtered_parse_structure)
                
                current_num_samples += 1
                if current_num_samples >= max_data_samples:
                    break
            idx += 1
    return words, parse_structures, sentence_lengths, max_sentence_length
    
    


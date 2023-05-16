### TAKEN FROM DANIEL BAUER'S NLP CODE ###

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    # helper function
    def valid_transition(self, action, state):
        buff, stack = state.buffer, state.stack
        if (action[0] == 'left_arc' or action[0] == 'right_arc') and len(stack) == 0:
            return False
        if action[0] == 'shift' and len(buff) == 1 and len(stack) > 0:
            return False
        if action[0] == 'left_arc' and len(stack) > 0 and stack[-1] == 0:
            return False
        return True
    
    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        ordered = []
        while state.buffer: 
                
            # TODO: Write the body of this loop for part 4
            
            input_repr = self.extractor.get_input_representation(words, pos, state)
            pred = self.model.predict(np.asarray([input_repr]))
            sorted_idxs = np.flip(np.argsort(pred)[0])
            
            curr_pos = 0
            while not self.valid_transition(self.output_labels[sorted_idxs[curr_pos]], state):
                curr_pos += 1
            if(words == [None, 'But', 'recent', 'developments', 'have', 'made', 'the', 'networks', '--', 'and', 'NBC', 'President', 'Robert', 'Wright', ',', 'in', 'particular', '--', 'ever', 'more', 'adamant', 'that', 'the', 'networks', 'must', 'be', 'unshackled', 'to', 'survive', '.']):
                print(ordered, state.stack, state.buffer)
            
            if self.output_labels[sorted_idxs[curr_pos]][0] == 'shift':
                ordered.append(('shift', 0))
                state.shift()
            
            elif self.output_labels[sorted_idxs[curr_pos]][0] == 'left_arc':
                ordered.append(('left_arc', self.output_labels[sorted_idxs[curr_pos]][1]))
                state.left_arc(self.output_labels[sorted_idxs[curr_pos]][1])
            
            elif self.output_labels[sorted_idxs[curr_pos]][0] == 'right_arc':
                ordered.append(('right_arc', self.output_labels[sorted_idxs[curr_pos]][1]))
                state.right_arc(self.output_labels[sorted_idxs[curr_pos]][1])
                 
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            #print(deps.print_conll())
            #print()
        

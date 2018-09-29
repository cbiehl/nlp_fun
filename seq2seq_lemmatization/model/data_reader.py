import numpy as np
np.random.seed(42)

import os

def load_dataset(filename, data_path="data", seq2seq=False):
    inflected_words = []
    lemmata = []

    with open(os.path.join(data_path, filename), 'r', encoding='utf8') as lines:
        # lists of characters of the inflected word and the lemma
        inflec = []
        lemma = []

        for line in lines:
            # empty line -> a word ends
            if not line.strip():
                if seq2seq:
                    ##########################################
                    #                                        #
                    #   maybe add your implementation here   #
                    #                                        #
                    ##########################################
                    for i, char in enumerate(lemma):
                        if char == 'EMPTY':
                            del lemma[i]
                        
                    for i, char in enumerate(lemma):
                        if '_MYJOIN_' in char:
                            splitchar = char.split('_MYJOIN_')
                            lemma[i] = splitchar[0]
                            lemma.append(splitchar[1])
                            
                    for i, char in enumerate(inflec):
                        if 'EMPTY' in char:
                            del inflec[i]
                            
                    for i, char in enumerate(inflec):
                        if '_MYJOIN_' in char:
                            splitchar = char.split('_MYJOIN_')
                            inflec[i] = splitchar[0]
                            inflec.append(splitchar[1])
                            
                    lemma.append('\n') #end of sequence character

                # store assembled inputs
                inflected_words.append(inflec)
                lemmata.append(lemma)
                inflec = []
                lemma = []
                lemma.append('\t') #start of sequence character
                continue

            inflec_char, lemma_char = line.strip().split('\t')
            inflec.append(inflec_char)

            if seq2seq:
                if lemma_char == 'EMPTY':
                    continue
                ##########################################
                #                                        #
                #   maybe add your implementation here   #
                #                                        #
                ##########################################
                lemma.append(lemma_char)
            else:
                lemma.append(lemma_char)

    return inflected_words, lemmata
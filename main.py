
import numpy as np
import random
import tensorflow as tf

# Importing File locally 
filepath = tf.keras.utils.get_file("shakespear.txt", '/Users/samakshtyagi/VS Code/Python/shakespear_text_generator /shakespeare.txt')

# Reading file and converting into numerical data for our neural network
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Preparing Data by selecting a limited part for training
text = text [300000:800000] #processing 500000 characters

# Creating unique set of text and Sorting it  
characters = sorted(set(text))

char_to_index = dict((c,i) for i,c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))


SEQ_LENGTH = 40 #Defining sequence length and 
STEP_SIZE = 3 #Defining step value size for next sentence.

sentences = []
next_char = []


# Iterating the entire text to gather the sentences and their next characters. 
# Training data for our neural network in textual form. (to be converted in numberical form)
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])  
    
# Creating 2 numpy arrays with zeroes
x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool)

# when a character appears at certain position we set it to 1
# dimensions: postion of sentence: postion within sentence : postion to specify which character 
for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1
    
    
    
#Data is prepared now. 

# next step : building recurrent neural network
# tbd

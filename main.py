import numpy as np
import random
import tensorflow as tf

from tenserflow.keras.model import Sequential #for our model
from tenserflow.keras.layers import LSTM, Dense, Activation  #for our layers
from tenserflow.keras.optimizers import RMSprop # for optimisation during the compilation o four model

# Importing File locally 
filepath = tf.keras.utils.get_file("shakespear.txt", '/Users/samakshtyagi/VS Code/Python/shakespear_text_generator /shakespeare.txt')

# Reading file and converting into numerical data for our neural network
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Preparing Data by selecting a limited part for training
text = text [300000:800000] #processing 500000 characters

# Creating unique set of text and Sorting it in place.
characters = sorted(set(text))

# Creating 2 dictionaries:
# 1. with character as key, index as value 
char_to_index = dict((c,i) for i,c in enumerate(characters))
# 2. with index as key and char as value.
index_to_char = dict((i,c) for i,c in enumerate(characters))


SEQ_LENGTH = 40 #Defining sequence length and 
STEP_SIZE = 3 #Defining step value size for next sentence.

''' 
if sentence is: "Hello World, I am a python user" 
it will scan the first <SEQ_LENGTH> (let's say 6) which is "HELL0 "...
and predict the next <STEP_SIZE> (let's say = 3) characters: ..."Wor"...
'''
sentences = [] #features 
next_char = [] 

# Iterating the entire text to gather the sentences and their next characters. 
# Training data for our neural network in textual form. (to be converted in numberical form later)
# this loop runs from beginning of text up until SEQ_LENGTH with a STEP_SIZE
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE): 
    sentences.append(text[i: i + SEQ_LENGTH]) #if SEQ_LENGTH is 5, so sentence is from 0-4 index and
    next_char.append(text[i + SEQ_LENGTH])   # next character is the one at 5th index.
    
# Creating 2 numpy arrays with zeroes
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)
# Whenever in a specific sentence, at a specific position, a speciifc char occurs, we will set it to True/1. all other values remain 0.

# dimensions: 1. for all the possible sentences that we have. 
#             2. for all the individual postions within these sentences
#             3. for all the possible characters that we can have   

for i, sentz in enumerate(sentences): #taking all sentences and assigning them an index
    for t, char in enumerate(sentz): #taking each sentence and assingning index to each character in each sentence
        x[i, t, char_to_index[char]] = 1
# i=sentence no. , t=pos no. , char 
    y[i, char_to_index[next_char[i]]] = 1
# the next character here is this one
    
    
#Data is prepared now. 

# next step : building recurrent neural network
# tbd

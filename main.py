
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


#tbd

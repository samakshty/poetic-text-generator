import certifi
import os

# Set the environment variable for SSL certificate file to certifi's CA bundle path
os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------MAIN CODE-------------------- #

import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential #for our model
from tensorflow.keras.layers import LSTM, Dense, Activation  #for our layers. LSTM is a type of RNN that is capable of learning long-term dependencies.
from tensorflow.keras.optimizers import RMSprop # for optimisation during the compilation o four model

# Importing File locally 
filepath = tf.keras.utils.get_file("shakespear.txt", 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

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

    # if sentence is: "Hello World, I am a python user" 
    # it will scan the first <SEQ_LENGTH> (let's say 6) which is "HELL0 "...
    # and predict the next <STEP_SIZE> (let's say = 3) characters: ..."Wor"...

sentences = [] #features 
next_char = [] 

# YOU WILL HAVE TO COMMENT OUT LINES 51-94 AFTER RUNNING THIS FILE ONCE, CUZ THE MODEL DOESN'T NEED TO BE TRAINED AGAIN.
    
    # Iterating the entire text to gather the sentences and their next characters. 
    # Training data for our neural network in textual form. (to be converted in numberical form later)
    # this loop runs from beginning of text up until SEQ_LENGTH with a STEP_SIZE
    
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE): 
    sentences.append(text[i: i + SEQ_LENGTH]) #if SEQ_LENGTH is 5, so sentence is from 0-4 index and
    next_char.append(text[i + SEQ_LENGTH])   # next character is the one at 5th index.
    
# Creating 2 numpy arrays with zeroes
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

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
    
# Data is prepared now. 

# Next step : building recurrent neural network:

model = Sequential()  # this initialized new instance of sequential model in keras
model.add(LSTM(128, input_shape = (SEQ_LENGTH, len(characters)))) 
    # The input will fall into our LSTM layer with 128 neurons. Input shape is length of sentence times the amount of character.
    # The character which shall follow will be set to True or False. 
model.add(Dense(len(characters))) 
    # Dense is a hidden layer, just to increase complexity. It has as many neurons as our possible '(characters)'
model.add(Activation('softmax')) 
    # Softmax Activation function scales the output so all values add up to one (100%). (This gives probability for each character. The output is alwasy of how likely a certain char is going to be the next character. Eg 'o' 70% then 'k will be 100-70=30%)

# Compiling Model now: 
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
    # loss function is categorical crossentropy, optimizer is RMSprop with learning rate of 0.01
    # RMSprop is a popular optimizer for RNNs. It adapts the learning rate for each

model.fit(x, y, batch_size=256, epochs=4) 
    #  we fitted the model on training data (x,y) , 
    # batch size = how many examples we are gonna put in the network at once. 
    # epoch = how many times our network will see the same data over and ober again 

model.save('textgenerator.keras')

# AFTER SAVING THE MODEL ONCE, COMMENT OUT LINES 51-94

# Loading the model
model = tf.keras.models.load_model('textgenerator.keras')

# Helper Function:
    
    # The sample function chooses the next character based on the softmax probabilities of each character.
    # It uses a 'temperature' parameter to control the randomness of the selection:
    # - High temperature -> more random, creative, and experimental choices.
    # - Low temperature -> safer, more conservative choices.
    # This function is utilized in the generate_text function, which generates text of a specified length using the given temperature.
    
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generating Text:
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1) # somewhere in the text we'll pick an entry point and take the first 40 characters.
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]  
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1 #like before, we put 1 wherever a char occurs

        predictions = model.predict(x_predictions, verbose=0)[0] # we get the softmax predictions
        next_index = sample(predictions, temperature) #feeding those in sample func with given temperature
        next_character = index_to_char[next_index] # converting it into next character

        generated += next_character  # adding the predicted char into the gen text so that we can use it as next input.
        sentence = sentence[1:] + next_character # left shifting the sentence and adding the new char in the end.
    return generated #finally returning the generated text

print('--------0.2-temp-------')
print(generate_text(300, 0.2))

print('--------0.4-temp-------')
print(generate_text(300, 0.4))

print('--------0.6-temp-------')
print(generate_text(300, 0.6))

print('--------0.8-temp-------')
print(generate_text(300, 0.8))

print('--------1.0-temp-------')
print(generate_text(300, 1.0))

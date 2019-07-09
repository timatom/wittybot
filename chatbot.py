# We'll need the following for our Natural Language Processing:
import nltk
from nltk.stem.lancaster import LancasterStemmer
stem = LancasterStemmer()

# We'll need the following for Tensorflow:
import numpy as np
import tflearn
import tensorflow as tf
import random


# This is to restore all of the data structures:
import pickle
data = pickle.load( open( "training_data", "rb" ))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Those intents file of ours, the following imports it:
import json
with open('conv_intents.json') as json_data:
    intents = json.load(json_data)


# Here we'll build our neural network as follows:
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# This is where we define our model, and then create our tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# The following will perform tokenization on detected patterns
# and then stem the words
def clean_up_sentence(sentence):
    # Tokenization
    sentence_words = nltk.word_tokenize(sentence)
    # Word stem
    sentence_words = [stem.stem(word.lower()) for word in sentence_words]
    return sentence_words

# The following will return the bag of words array:
# 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # Tokenization
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# The following will load the previously saved model (the model created from
# executing our model.py file):
model.load('./model.tflearn')

# The following creates a data structure to hold user context:
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # Generates probabilities from the model:
    results = model.predict([bow(sentence, words)])[0]
    # Filters out predictions below a threshold:
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # Sorts by strength of probability:
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # Return tuple of intent and probability:
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # The following finds a matching intent tag with an existing classification:
    if results:
        # Iterate through all intents until there are no more matches to process:
        while results:
            for i in intents['intents']:
                # Find a tag matching the first result:
                if i['tag'] == results[0][0]:
                    # If necessary, set context for this intent:
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # Check if this intent is contextual and applies to this user's conversation:
                    if not 'context_filter' in i or \
                    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # The following prints a random response associated with the intent:
                        return print(random.choice(i['responses']))

            results.pop(0)

# Defines a status variable:
stat = "g"

# Continue to request a user response until they enter 'q' to quit the chat:
while (stat != "q"):
    user_input = input("Input (q to quit): ")
    if (user_input == "q"):
        stat = "q"
        pass

    response(user_input)

# We'll need the following for our Natural Language Processing
import nltk
from nltk.stem.lancaster import LancasterStemmer
stem = LancasterStemmer()

# We'll need the following for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


# The following imports our json and loads the json data:
import json
with open('conv_intents.json') as json_data:
    intents = json.load(json_data)


nltk.download('punkt')
words = []
classes = []
documents = []
ignore_words = ['?']
# Iterate through each sentence in the intents:
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenization of words
        w = nltk.word_tokenize(pattern)
        # Adding to words list:
        words.extend(w)
        # Adding to documents in corpus:
        documents.append((w, intent['tag']))
        # Adding to classes list:
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and lower each word and remove duplicates:
words = [stem.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Remove duplicates:
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


# Create our training data:
training = []
output = []

# Create an empty array for our output:
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence:
for doc in documents:
    # Initializes bag of words:
    bag = []
    # List of tokenized words for pattern:
    pattern_words = doc[0]
    # Stem each word:
    pattern_words = [stem.stem(word.lower()) for word in pattern_words]
    # Create bag of words array:
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Assigns '0' for each tag and '1' for current tag:
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffles features and turns into np.array:
random.shuffle(training)
training = np.array(training)

# Creates train and test lists:
train_x = list(training[:,0])
train_y = list(training[:,1])


# Resets underlying graph data:
tf.reset_default_graph()
# Builds neural network:
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defines model and creates tensorboard:
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Starts training (Gradient Descent Algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


# Saves all of our data structures:
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

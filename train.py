import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import random
import json
import pickle

# Load intents
with open("intents.json") as f:
    data = json.load(f)

words = []
labels = []
docs_x = []
docs_y = []

# read patterns from intents.json
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem & sort words
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# Create training data
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        bag.append(1 if w in wrds else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Save processed data
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(len(output[0]), activation="softmax")
])

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train model
model.fit(training, output, epochs=200, batch_size=8)

# Save trained model
model.save("model.h5")

print("Training complete. Model saved as model.h5")

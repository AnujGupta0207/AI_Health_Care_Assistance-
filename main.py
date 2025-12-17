import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import random
import json
import pickle

from symptom_engine import predict_diseases  # ⬅ NEW IMPORT

# Load intents
with open("intents.json") as f:
    data = json.load(f)

# Load training data
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model("model.h5")

def bag_of_words(sentence, words):
    bag = [0 for _ in words]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Healthcare Assistant is ready! Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        inp_clean = inp.lower().strip()

        if inp_clean == "quit":
            print("Bot: Thank you for using the Healthcare Assistant. Take care! ❤️")
            break

        # 1️⃣ If user listed symptoms (comma-separated), use symptom engine
        if "," in inp_clean:
            possible = predict_diseases(inp_clean)

            if not possible:
                print("Bot: I couldn't clearly match your symptoms to specific conditions.")
                print("Bot: Please consult a doctor or visit a nearby clinic for proper diagnosis.")
            else:
                msg = "Based on your symptoms, possible conditions are:\n"
                for disease, score in possible:
                    msg += f"- {disease.replace('_', ' ').title()} (matched {score} symptom(s))\n"
                msg += "\n⚠ This is NOT a confirmed medical diagnosis.\nPlease consult a doctor for accurate examination and tests."
                print("Bot:", msg)
            continue  # skip normal intent prediction for this loop

        # 2️⃣ Normal intent-based chatbot (greeting, goodbye, symptom_check, etc.)
        bow = bag_of_words(inp, words)
        results = model.predict(np.array([bow]))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Lowered confidence threshold
        if results[results_index] > 0.4:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    break
            print("Bot:", random.choice(responses))
        else:
            print("Bot: I'm not sure I understood that. If you have symptoms, please list them using commas (e.g. fever, cough, headache).")

if __name__ == "__main__":
    chat()

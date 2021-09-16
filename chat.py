import nltk
import numpy as np
import torch
import torch.nn as nn
import random
import json                                
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data['allWords']
tags = data['tags']
modelState = data["modelState"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)
model.eval()

botName = "Tom"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bagOfWords(sentence, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: I couldn't understand :(")
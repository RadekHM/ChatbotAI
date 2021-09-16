import nltk
import numpy as np
import torch
import torch.nn as nn
import random
import json                                
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize) 
        self.l2 = nn.Linear(hiddenSize, hiddenSize) 
        self.l3 = nn.Linear(hiddenSize, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
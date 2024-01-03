import time
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataset import PaddedTensorDataset
from dataset import TextLoader
from model_architecture import LSTMClassifier
from train_and_evaluation import train_model, evaluate_test_set, evaluate_validation_set

data_dir='Surnames'
hidden_dim=32           # can be tuned
batch_size=32
num_epochs=15
char_dim=128
learning_rate=0.01
weight_decay=1e-4
seed=12

random.seed(seed)
# NEED A MORE CAREFUL DIVISION IN TRAIN AND TEST DATASET
data_loader = TextLoader(data_dir)

# list of tuples made as ('Hori', 'Japanese')
train_data = data_loader.train_data
dev_data = data_loader.dev_data
test_data = data_loader.test_data

#defaultdict() of characters
char_vocab = data_loader.token2id
char_vocab_size = len(char_vocab)

#defaultdict() of languages
tag_vocab = data_loader.tag2id


print('Training samples:', len(train_data))
print('Valid samples:', len(dev_data))
print('Test samples:', len(test_data))

# we maintain a dictionary with all characters used
print(char_vocab)
# we keep a dictionary with all languages and their respective index
print(tag_vocab)

model = LSTMClassifier(char_vocab_size, char_dim, hidden_dim=hidden_dim, 
                       output_size=len(tag_vocab))
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

model = train_model(model, optimizer, train_data, dev_data, char_vocab, tag_vocab, 
                    batch_size, num_epochs)

evaluate_test_set(model, test_data, char_vocab, tag_vocab)
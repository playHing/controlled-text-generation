
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from fast_data import FastData

from ctextgen.dataset import *

import argparse


parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()

batch_size = 32
fast_data = FastData(data_type='test', batch_size=batch_size)
print('train data size: {}'.format(len(fast_data.train_data)))
vocab = fast_data.vocab

mb_size = 32
z_dim = 100
h_dim = 128
lr = 1e-3
lr_decay_every = 1000000
n_iter = 50000
log_interval = 200
z_dim = h_dim

dataset = SST_Dataset()
# dataset = WikiText_Dataset()
# dataset = IMDB_Dataset()


class Clf(nn.Module):

    def __init__(self):
        super(Clf, self).__init__()

        emb_dim = dataset.get_vocab_vectors().size(1)
        self.word_emb = nn.Embedding(dataset.n_vocab, emb_dim)
        # Set pretrained embeddings
        self.word_emb.weight.data.copy_(dataset.get_vocab_vectors())
        self.word_emb.weight.requires_grad = False

        self.conv3 = nn.Conv2d(1, 100, (3, emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, emb_dim))

        self.discriminator = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, 2)
        )

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(self, inputs):
        inputs = self.word_emb(inputs)
        inputs = inputs.unsqueeze(1)

        x3 = F.relu(self.conv3(inputs)).squeeze()
        x4 = F.relu(self.conv4(inputs)).squeeze()
        x5 = F.relu(self.conv5(inputs)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.discriminator(x)

        return y


model = Clf()
model.load_state_dict(torch.load('models/clf.bin'))
print('clf model loaded.')

accs = []

# Test on validation
while len(accs) < 9000:
    inputs, labels = fast_data.next_batch()
    inputs = inputs['word_seq']
    labels = labels['label_seq']

    res = model.forward(inputs)
    _, y = res.max(dim=1)

    acc = float((y == labels).sum()) / y.size(0)
    accs.append(acc)

    if len(accs) % 1000 == 0:
        print('Iter-{}; val_acc: {:.4f}'.format(len(accs), np.mean(accs)))

print('done.')

print('sst:')
accs = []

# Test on validation
for it in range(51):
    inputs, labels = dataset.next_validation_batch(args.gpu)
    inputs = inputs.transpose(0, 1)

    _, y = model.forward(inputs).max(dim=1)

    acc = float((y == labels).sum()) / y.size(0)
    accs.append(acc)

    if it % 10 == 0:
        print('Iter-{}; val_acc: {:.4f}'.format(it, np.mean(accs)))

print('done.')

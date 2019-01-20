import os
import math

import torch.optim as optim
from fast_model import *
from fast_data import FastData
from fast_config import FastConfig
import torch.nn.functional as F
from fast_ctrl_gen import *

from fastNLP.models import cnn_text_classification
from fastNLP import Trainer, Adam, LossFunc, AccuracyMetric

vae.load_state_dict(torch.load('models/vae-sst-updated.bin', map_location='cpu'))
disc.load_state_dict(torch.load('models/disc-sst.bin', map_location='cpu'))

fpos = open("fast_gen_data2.pos", "w")
fneg = open("fast_gen_data2.neg", "w")

epoch = 10000
from random import randint

c = sample_c_prior(cf.c_dim)

c[0, 0], c[0, 1] = 1, 0
for i in range(epoch):
    # Samples latent and conditional codes randomly from prior
    #z = sample_z_prior(cf.z_dim)

    cand = fast_data.train_data[randint(0, len(fast_data.train_data) - 1)]
    if cand['label_seq'] == 1:
        c[0, 0], c[0, 1] = 1, 0
    else:
        c[0, 0], c[0, 1] = 0, 1
    input = torch.LongTensor([cand['word_seq']])

    mu, log_var = encoder.forward(word_emb.forward(input))
    z = sample_z(mu, log_var)
    sample_idxs = sample_sentence(z, c, temp=1)
    sample_sent = ' '.join([vocab.to_word(idx) for idx in sample_idxs])

    if cand['label_seq'] == 1:
        fpos.write(sample_sent + '\n')
    else:
        fneg.write(sample_sent + '\n')

    if i % 500 == 0:
        print(i)

print('done.')

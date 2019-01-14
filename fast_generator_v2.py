import os
import math

import torch
import torch.optim as optim
from ctextgen.fast_model_v2 import *
from ctextgen.fast_data import FastData
from ctextgen.fast_config import FastConfig
import torch.nn.functional as F

from fastNLP.models import cnn_text_classification
from fastNLP import Trainer, Adam, LossFunc, AccuracyMetric

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c). '
                                'Fork and re-implement with fastNLP.')
parser.add_argument('--gpu', default=False, action='store_true', help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
args = parser.parse_args()

batch_size = 32
fast_data = FastData(batch_size=batch_size)
print('train data size: {}'.format(len(fast_data.train_data)))
vocab = fast_data.vocab

cf = FastConfig(h_dim=64, z_dim=64, c_dim=2, n_vocab=len(vocab),
                unk_idx=vocab.unknown_idx, pad_idx=vocab.padding_idx,
                start_idx=vocab.to_index('<start>'),
                eos_idx=vocab.to_index('<eos>'),
                gpu=args.gpu)

word_emb = WordEmbedding(cf)
encoder = Encoder(cf)
decoder = Decoder(word_emb, cf)
vae = VariationalAE(word_emb, encoder, decoder, cf)
disc = cnn_text_classification.CNNText(cf.n_vocab, cf.emb_dim, cf.c_dim)


def train_variational_ae():

    n_epochs = 100
    VariationalAE.init_hyper_params(n_epochs, -2)

    trainer = Trainer(
        train_data=fast_data.train_data,
        dev_data=None,
        model=vae,
        loss=LossFunc(VariationalAE.loss),
        metrics=None,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=Adam(lr=1e-3, weight_decay=0, model_params=vae.params)
    )
    trainer.train()
    print('Train finished!')


def save_model(path):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(vae.state_dict(), path)


if __name__ == '__main__':
    try:
        train_variational_ae()
        if args.save:
            save_model('models/vae.bin')
    except KeyboardInterrupt:
        exit(0)

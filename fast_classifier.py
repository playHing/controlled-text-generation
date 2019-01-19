import os
import math

import torch.optim as optim
from fast_model import *
from fast_data import FastData
from fast_config import FastConfig
import torch.nn.functional as F

from fastNLP.models import cnn_text_classification
from fastNLP import Trainer, Adam, AccuracyMetric, CrossEntropyLoss

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Fast classifier')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
args = parser.parse_args()

batch_size = 32
fast_data = FastData(data_type='test', batch_size=batch_size)
print('train data size: {}'.format(len(fast_data.train_data)))
vocab = fast_data.vocab

cf = FastConfig(emb_dim=64, h_dim=64, z_dim=64, c_dim=2, n_vocab=len(vocab),
                unk_idx=vocab.unknown_idx, pad_idx=vocab.padding_idx,
                start_idx=vocab.to_index('<start>'),
                eos_idx=vocab.to_index('<eos>'),
                gpu=False)

disc = cnn_text_classification.CNNText(cf.n_vocab, cf.emb_dim, cf.c_dim)


def train_classifier():
    n_epochs = 50

    trainer = Trainer(
        train_data=fast_data.train_data,
        dev_data=fast_data.test_data,
        model=disc,
        loss=CrossEntropyLoss(target='label_seq'),
        metrics=AccuracyMetric(target='label_seq'),
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=Adam(lr=0.001, weight_decay=0, model_params=disc.parameters())
    )
    trainer.train()
    print('Disc Train finished!')


def save_model(model: nn.Module, path):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    try:
        train_classifier()

        if args.save:
            save_model(disc, 'models/disc-test.bin')
            print('disc-test model saved.')

    except KeyboardInterrupt:
        exit(0)

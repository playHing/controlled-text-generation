import os
import io

import torch
import torch.optim as optim
from ctextgen.fast_model import RNN_VAE

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler

import argparse


parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c). Fork and reimplement with fastNLP.'
)
parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()

def prepare_data(path, dataset="yelp"):

    if dataset == "yelp":
        dataset = DataSet()

        for db_set in ['train']:
            text_file = os.path.join(path, 'sentiment.'+db_set+'.text')
            label_file = os.path.join(path, 'sentiment.'+db_set+'.labels')
            with io.open(text_file, 'r', encoding="utf-8") as tf, io.open(label_file, 'r', encoding="utf-8") as lf:
                for text in tf:
                    label = lf.readline()
                    dataset.append(Instance(text=text, label=label))

        dataset.apply(lambda x: x['text'].lower(), new_field_name='text')
        dataset.apply(lambda x: ['<start>']+x['text'].split()+['<eos>'], new_field_name='words')
        dataset.drop(lambda x: len(x['words']) > 1+15+1)
        dataset.apply(lambda x: x['words']+['<pad>']*(17-len(x['words'])), new_field_name='words')
        dataset.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

        print(dataset.get_length())

        _train_data, _test_data = dataset.split(0.3)

        vocab = Vocabulary(min_freq=2)
        _train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
        vocab.build_vocab()

        _train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
        _test_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)

        return _train_data, _test_data, vocab


batch_size = 32
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 2

train_data, test_data, vocab = prepare_data(".data/yelp")
print(train_data.get_length())
train_iter = Batch(dataset=train_data, batch_size=batch_size, sampler=SequentialSampler())
train_iter = iter(train_iter)

model = RNN_VAE(
    len(vocab), h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=None, freeze_embeddings=False,
    unk_idx=vocab.unknown_idx, pad_idx=vocab.padding_idx,
    start_idx=vocab.to_index('<start>'), eos_idx=vocab.to_index('<eos>'),
    gpu=args.gpu
)

def train_vae():
    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

    trainer = optim.Adam(model.vae_params, lr=lr)

    for it in range(n_iter):

        inputs, labels = next(train_iter)

        recon_loss, kl_loss = model.forward(torch.t(inputs['word_seq'])) # (17, 32)
        loss = recon_loss + kld_weight * kl_loss

        # Anneal kl_weight
        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = ' '.join(list(map(vocab.to_word, sample_idxs)))

            print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f};'
                  .format(it, loss.data.item(), recon_loss.data.item(), kl_loss.data.item(), grad_norm))

            print('Sample: "{}"'.format(sample_sent))
            print()

        # Anneal learning rate
        new_lr = lr * (0.5 ** (it // lr_decay_every))
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/vae.bin')


if __name__ == '__main__':
    try:
        train_vae()
    except KeyboardInterrupt:
        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()

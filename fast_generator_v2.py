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

cf = FastConfig(emb_dim=64, h_dim=64, z_dim=64, c_dim=2, n_vocab=len(vocab),
                unk_idx=vocab.unknown_idx, pad_idx=vocab.padding_idx,
                start_idx=vocab.to_index('<start>'),
                eos_idx=vocab.to_index('<eos>'),
                gpu=args.gpu)

disc = cnn_text_classification.CNNText(cf.n_vocab, cf.emb_dim, cf.c_dim)
word_emb = disc.embed
encoder = Encoder(cf)
decoder = Decoder(word_emb, cf)
vae = VariationalAE(word_emb, encoder, decoder, cf)


def train_variational_ae():
    n_epochs = 5

    trainer = Trainer(
        train_data=fast_data.train_data,
        dev_data=fast_data.test_data,
        model=vae,
        loss=LossFunc(VariationalAE.loss),
        metrics=AccuracyMetric(target='dec_target'),
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer=Adam(lr=0.001, weight_decay=0, model_params=vae.parameters())
    )
    trainer.train()
    print('VAE Train finished!')


def generate_sentences():
    """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
    """
    samples = []
    cs = []

    for _ in range(batch_size):
        z = sample_z_prior(cf.z_dim)
        c = sample_c_prior(cf.c_dim)
        samples.append(sample_sentence(z, c, raw=True))
        cs.append(c.long())

    x_gen = torch.cat(samples, dim=0)
    c_gen = torch.cat(cs, dim=0)

    return x_gen, c_gen


def sample_sentence(z, c, raw=False, temp=1):
    """
    Sample single sentence from p(x|z,c) according to given temperature.
    `raw = True` means this returns sentence as in dataset which is useful
    to train discriminator. `False` means that this will return list of
    `word_idx` which is useful for evaluation.
    """

    word = torch.LongTensor([cf.START_IDX])
    word = word.cuda() if cf.gpu else word
    word = Variable(word)  # '<start>'

    z, c = z.view(1, 1, -1), c.view(1, 1, -1)
    h = torch.cat([z, c], dim=2)

    if not isinstance(h, Variable):
        h = Variable(h)

    outputs = []

    if raw:
        outputs.append(cf.START_IDX)

    for i in range(cf.MAX_SENT_LEN):
        emb = word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z, c], 2)

        output, h = decoder.decoder(emb, h)
        y = decoder.decoder_fc(output).view(-1)
        y = F.softmax(y/temp, dim=0)

        idx = torch.multinomial(y, 1)

        word = Variable(torch.LongTensor([int(idx)]))
        word = word.cuda() if cf.gpu else word

        idx = int(idx)

        if not raw and idx == cf.EOS_IDX:
            break

        outputs.append(idx)

    if raw:
        outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
        return outputs.cuda() if cf.gpu else outputs
    else:
        return outputs


def generate_soft_embed(temp=1.):
    """
    Generate soft embeddings of (batch_size x emb_dim) along with target z
    and c for each row (batch_size x {z_dim, c_dim})
    """
    samples = []
    targets_c = []
    targets_z = []

    for _ in range(batch_size):
        z = sample_z_prior(cf.z_dim)
        c = sample_c_prior(cf.c_dim)

        samples.append(sample_soft_embed(z, c, temp=temp))
        targets_z.append(z)
        targets_c.append(c)

    x_gen = torch.cat(samples, dim=0)
    targets_z = torch.cat(targets_z, dim=0)
    _, targets_c = torch.cat(targets_c, dim=0).max(dim=1)

    return x_gen, targets_z, targets_c


def sample_soft_embed(z, c, temp=1.):
    """
    Sample single soft embedded sentence from p(x|z,c) and temperature.
    Soft embeddings are calculated as weighted average of word_emb
    according to p(x|z,c).
    """

    word = torch.LongTensor([cf.START_IDX])
    word = word.cuda() if cf.gpu else word
    word = Variable(word)  # '<start>'

    z, c = z.view(1, 1, -1), c.view(1, 1, -1)
    h = torch.cat([z, c], dim=2)

    if not isinstance(h, Variable):
        h = Variable(h)

    outputs = [word_emb(word).view(1, -1)]

    emb = word_emb(word).view(1, 1, -1)
    emb = torch.cat([emb, z, c], 2)

    for i in range(cf.MAX_SENT_LEN):
        output, h = decoder.decoder(emb, h)
        o = decoder.decoder_fc(output).view(-1)

        # Sample softmax with temperature
        y = F.softmax(o / temp, dim=0)

        # Take expectation of embedding given output prob -> soft embedding
        # <y, w> = 1 x n_vocab * n_vocab x emb_dim
        emb = y.unsqueeze(0) @ word_emb.embed.weight
        emb = emb.view(1, 1, -1)

        # Save resulting soft embedding
        outputs.append(emb.view(1, -1))

        # Append with z and c for the next input
        emb = torch.cat([emb, z, c], 2)

    # 1 x 16 x emb_dim
    outputs = torch.cat(outputs, dim=0).unsqueeze(0)

    return outputs.cuda() if cf.gpu else outputs


def kl_weight(it):
    """
    Credit to: https://github.com/kefirski/pytorch_RVAE/
    0 -> 1
    """
    return (math.tanh((it - 3500)/1000) + 1)/2


def temp(it):
    """
    Softmax temperature annealing
    1 -> 0
    """
    return 1-kl_weight(it) + 1e-5  # To avoid overflow


def train_ctrl_gen():

    lr = 0.001
    n_iter = 5
    log_interval = 100

    beta = 0.1
    lambda_c = 0.1
    lambda_z = 0.1
    lambda_u = 0.1

    trainer_d = optim.Adam(disc.parameters(), lr=lr)
    trainer_g = optim.Adam(decoder.parameters(), lr=lr)
    trainer_e = optim.Adam(encoder.parameters(), lr=lr)

    for it in tqdm(range(n_iter)):

        inputs, labels = fast_data.next_batch()

        """ Update disc, eq. 11 """
        # get sentences and corresponding z
        x_gen, c_gen = generate_sentences()
        _, target_c = torch.max(c_gen, dim=1)

        y_disc_real = disc.forward(inputs['word_seq'])['pred']
        y_disc_fake = disc.forward(x_gen)['pred']

        log_y_disc_fake = F.log_softmax(y_disc_fake, dim=1)
        entropy = -log_y_disc_fake.mean()

        loss_s = F.cross_entropy(y_disc_real, labels['label_seq'])
        loss_u = F.cross_entropy(y_disc_fake, target_c) + beta * entropy

        loss_d = loss_s + lambda_u * loss_u

        loss_d.backward()
        torch.nn.utils.clip_grad_norm(disc.parameters(), 5)
        trainer_d.step()
        trainer_d.zero_grad()

        """ Update generator, eq. 8 """
        # Forward VAE with c ~ q(c|x) instead of from prior
        disc_pred = disc.forward(inputs['word_seq'])['pred']
        vae_output = vae.forward_with_disc_pred(inputs['word_seq'], disc_pred)

        # x_gen: batch_size x seq_len x emb_dim
        x_gen_attr, target_z, target_c = generate_soft_embed(temp=temp(it))

        # y_z: batch_size x z_dim
        y_z, _ = encoder.forward(x_gen_attr)
        y_c = disc.fc(disc.dropout(disc.conv_pool(x_gen_attr)))

        loss_vae = VariationalAE.loss(vae_output['pred'], vae_output['mu'], vae_output['log_var'], labels['dec_target'])
        loss_attr_c = F.cross_entropy(y_c, target_c)
        loss_attr_z = F.mse_loss(y_z, target_z)

        loss_g = loss_vae + lambda_c * loss_attr_c + lambda_z * loss_attr_z

        loss_g.backward()
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 5)
        trainer_g.step()
        trainer_g.zero_grad()

        """ Update encoder, eq. 4 """
        disc_pred = disc.forward(inputs['word_seq'])['pred']
        vae_output = vae.forward_with_disc_pred(inputs['word_seq'], disc_pred)

        loss_e = VariationalAE.loss(vae_output['pred'], vae_output['mu'], vae_output['log_var'], labels['dec_target'])

        loss_e.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), 5)
        trainer_e.step()
        trainer_e.zero_grad()

        if it % log_interval == 0:
            z = sample_z_prior(cf.z_dim)
            c = sample_c_prior(cf.c_dim)

            sample_idxs = sample_sentence(z, c)
            sample_sent = [vocab.to_word(idx) for idx in sample_idxs]

            print('Iter-{}; loss_d: {:.4f}; loss_G: {:.4f}'
                  .format(it, float(loss_d), float(loss_g)))

            _, c_idx = torch.max(c, dim=1)

            print('c = {}'.format(int(c_idx)))
            print('Sample: "{}"'.format(sample_sent))
            print()


def save_model(path):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(vae.state_dict(), path)


if __name__ == '__main__':
    try:
        #train_variational_ae()
        if args.save:
            save_model('models/vae.bin')
        train_ctrl_gen()

    except KeyboardInterrupt:
        exit(0)

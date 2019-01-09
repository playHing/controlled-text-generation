import os
import math

import torch
import torch.optim as optim
from ctextgen.fast_model_v2 import AutoEncoder
from ctextgen.fast_data import FastData
from ctextgen.fast_config import FastConfig
import torch.nn.functional as F

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c). '
                                'Fork and reimplement with fastNLP.')
parser.add_argument('--gpu', default=False, action='store_true', help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
args = parser.parse_args()

fast_data = FastData()
print('train data size: {}'.format(len(fast_data.train_data)))
vocab = fast_data.vocab

cf = FastConfig(h_dim=64, z_dim=64, c_dim=2, n_vocab=len(vocab),
                unk_idx=vocab.unknown_idx, pad_idx=vocab.padding_idx,
                start_idx=vocab.to_index('<start>'),
                eos_idx=vocab.to_index('<eos>'),
                gpu=args.gpu)

model = AutoEncoder(cf)

def train_auto_encoder(n_iter=20000, log_interval=1000):

    lr = 1e-3
    lr_decay_every = 1000000

    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

    trainer = optim.Adam(model.vae_params, lr=lr)

    for it in range(n_iter):

        inputs, _ = fast_data.next_batch()
        inputs = inputs['word_seq']  # (32, 17)

        recon_loss, kl_loss = model.compute_loss_vae(inputs)
        loss = recon_loss + kld_weight * kl_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = ' '.join(list(map(vocab.to_word, sample_idxs)))

            print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f}; lr: {:.4f};'
                  .format(it, loss.data.item(), recon_loss.data.item(), kl_loss.data.item(), grad_norm, lr))
            print('Sample: "{}"'.format(sample_sent))
            print()

        # Anneal kl_weight
        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        # Anneal learning rate
        lr = lr * (0.5 ** (it // lr_decay_every))
        for param_group in trainer.param_groups:
            param_group['lr'] = lr


def train_generator(n_iter=5000, log_interval=100):
    lr = 1e-3
    lr_decay_every = 1000000
    kl_weight_max = 0.4

    # Specific hyperparams
    beta = 0.1
    lambda_c = 0.1
    lambda_z = 0.1
    lambda_u = 0.1

    trainer_D = optim.Adam(model.discriminator_params, lr=lr)
    trainer_G = optim.Adam(model.decoder_params, lr=lr)
    trainer_E = optim.Adam(model.encoder_params, lr=lr)

    for it in tqdm(range(n_iter)):

        inputs, labels = fast_data.next_batch()
        inputs = inputs['word_seq']  # (32, 17)
        labels = labels['label_seq']

        """ Update discriminator, eq. 11 """
        # get sentences and corresponding z
        x_gen, c_gen = model.generate_sentences()
        _, target_c = torch.max(c_gen, dim=1)

        y_disc_real = model.forward_discriminator(inputs)
        y_disc_fake = model.forward_discriminator(x_gen)

        log_y_disc_fake = F.log_softmax(y_disc_fake, dim=1)
        entropy = -log_y_disc_fake.mean()

        loss_s = F.cross_entropy(y_disc_real, labels)
        loss_u = F.cross_entropy(y_disc_fake, target_c) + beta * entropy
        loss_D = loss_s + lambda_u * loss_u
        loss_D.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
        trainer_D.step()
        trainer_D.zero_grad()

        """ Update generator, eq. 8 """
        # Forward VAE with c ~ q(c|x) instead of from prior
        recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)
        # x_gen: mbsize x seq_len x emb_dim
        x_gen_attr, target_z, target_c = model.generate_soft_embed(temp=temp(it))

        # y_z: mbsize x z_dim
        y_z, _ = model.forward_encoder_embed(x_gen_attr.transpose(0, 1))
        y_c = model.forward_discriminator_embed(x_gen_attr)

        loss_vae = recon_loss + kl_weight_max * kl_loss
        loss_attr_c = F.cross_entropy(y_c, target_c)
        loss_attr_z = F.mse_loss(y_z, target_z)

        loss_G = loss_vae + lambda_c * loss_attr_c + lambda_z * loss_attr_z

        loss_G.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.decoder_params, 5)
        trainer_G.step()
        trainer_G.zero_grad()

        """ Update encoder, eq. 4 """
        recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)

        loss_E = recon_loss + kl_weight_max * kl_loss

        loss_E.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.encoder_params, 5)
        trainer_E.step()
        trainer_E.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = ' '.join(list(map(vocab.to_word, sample_idxs)))

            print('Iter-{}; loss_D: {:.4f}; loss_G: {:.4f}'
                  .format(it, float(loss_D), float(loss_G)))

            _, c_idx = torch.max(c, dim=1)

            print('c = {}'.format(c_idx))
            print('Sample: "{}"'.format(sample_sent))
            print()


def kl_weight(it):
    """
    Credit to: https://github.com/kefirski/pytorch_RVAE/
    0 -> 1
    """
    return (math.tanh((it - 3500) / 1000) + 1) / 2


def temp(it):
    """
    Softmax temperature annealing
    1 -> 0
    """
    return 1 - kl_weight(it) + 1e-5  # To avoid overflow


def save_model(path):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    try:

        train_vae()
        if args.save:
            save_model('models/vae.bin')

        #train_generator()
        #if args.save:
        #    save_model('models/ctextgen.bin')

    except KeyboardInterrupt:
        exit(0)

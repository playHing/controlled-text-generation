import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain

from .fast_config import FastConfig
from fastNLP import models, modules


class WordEmbedding(nn.Module):

    def __init__(self, cf: FastConfig):
        super(WordEmbedding, self).__init__()
        if cf.pre_trained_embeddings is None:
            self.emb_dim = cf.h_dim
            self.embedding = nn.Embedding(cf.n_vocab, self.emb_dim, cf.PAD_IDX)
        else:
            self.emb_dim = cf.pre_trained_embeddings.size(1)
            self.embedding = nn.Embedding(cf.n_vocab, self.emb_dim, cf.PAD_IDX)
            self.embedding.weight.data.copy_(cf.pre_trained_embeddings)        # Set pre-trained embeddings
            if cf.freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # set embedding dimension on configuration
        cf.emb_dim = self.emb_dim

    def forward(self, word_seq):
        """
        :param word_seq: [batch_size, seq_len]
        :return: [batch_size, seq_len, emb_dim]
        """
        return self.embedding(word_seq)


class Encoder(nn.Module):

    def __init__(self, cf: FastConfig):
        """
            Encoder is GRU with FC layers connected to last hidden unit
        """
        super(Encoder, self).__init__()
        self.encoder = nn.GRU(cf.emb_dim, cf.h_dim)
        self.q_mu = nn.Linear(cf.h_dim, cf.z_dim)           # mean
        self.q_log_var = nn.Linear(cf.h_dim, cf.z_dim)      # variance
        self.cf = cf

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: [batch_size, seq_len, emb_dim]
        :return: mu, log_var, [batch_size, z_dim]
        """
        _, h = self.encoder(inputs, None)                   # h is the last output of GRU

        # Forward to latent
        h = h.view(-1, self.cf.h_dim)
        mu = self.q_mu(h)
        log_var = self.q_log_var(h)
        return mu, log_var


class Decoder(nn.Module):

    def __init__(self, word_emb: WordEmbedding, cf: FastConfig):
        super(Decoder, self).__init__()
        self.word_emb = word_emb
        self.decoder = nn.GRU(cf.emb_dim + cf.z_dim + cf.c_dim, cf.z_dim + cf.c_dim, dropout=0.3)
        self.decoder_fc = nn.Linear(cf.z_dim + cf.c_dim, cf.n_vocab)
        self.cf = cf

    def forward(self, word_seq, z, c):
        """
        :param word_seq: [batch_size, seq_len]
        :param z: [batch_size, z_dim]
        :param c: [batch_size, c_dim]
        :return: [batch_size, seq_len, n_vocab]
        """

        dec_inputs = self.word_dropout(word_seq)

        # Forward
        seq_len = dec_inputs.size(0)
        init_h = torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)             # [batch_size, z_dim+c_dim]
        inputs_emb = self.word_emb(dec_inputs)                                  # [batch_size, seq_len, emb_dim]
        init_h_repeat = init_h.repeat(seq_len, 1, 1)
        inputs_emb = torch.cat([inputs_emb, init_h_repeat], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, batch_size, _ = outputs.size()

        outputs = outputs.view(seq_len * batch_size, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, batch_size, self.cf.n_vocab)

        return y

    def word_dropout(self, inputs):
        """
            Do word dropout: with prob `dropout`, set the word to '<unk>'.
        """

        data = inputs.data.clone() if isinstance(inputs, Variable) else inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.cf.word_dropout_prob,
                               size=tuple(data.size())).astype('uint8')
            )

        # Set to <unk>
        data[mask] = self.cf.UNK_IDX

        return Variable(data)


def sample_z(mu: torch.Tensor, log_var: torch.Tensor):
    """
        Re-parametrization trick: z = mu + std * eps, eps ~ N(0, I)
    """
    assert mu.shape == log_var.shape, 'mu and log_var must have the same shape'

    z_dim = mu.shape[1]
    eps = Variable(torch.randn(z_dim))
    return mu + torch.exp(log_var / 2) * eps


def sample_z_prior(z_dim, size=1):
    """
        Sample z ~ p(z) = N(0, I)
    """
    z = Variable(torch.randn(size, z_dim))
    return z


def sample_c_prior(c_dim: int, size=1):
    """
        Sample c ~ p(c) = multinomial([1./c_dim] * c_dim)
    """
    assert c_dim > 0, 'c_dim must be > 0'

    c = Variable(
        torch.from_numpy(np.random.multinomial(1, [1.0/c_dim] * c_dim, size).astype('float32'))
    )
    return c


class VariationalAE(nn.Module):

    # word-around
    # Annealing for KL term
    kld_start_inc = None
    kld_weight = None
    kld_max = None
    n_iter = None
    kld_inc = None
    it = None

    def __init__(self, word_emb: WordEmbedding, encoder: Encoder,
                 decoder: Decoder, cf: FastConfig):
        super(VariationalAE, self).__init__()
        self.word_emb = word_emb
        self.encoder = encoder
        self.decoder = decoder
        self.cf = cf

        params = chain(
            self.word_emb.embedding.parameters(),
            self.encoder.encoder.parameters(),
            self.encoder.q_mu.parameters(),
            self.encoder.q_log_var.parameters(),
            self.decoder.decoder.parameters(),
            self.decoder.decoder_fc.parameters()
        )
        self.params = filter(lambda p: p.requires_grad, params)

    def forward(self, word_seq: torch.Tensor):
        self.train()
        word_seq.t_()       # plz be aware of the shape of the tensors

        embed = self.word_emb(word_seq)
        mu, log_var = self.encoder(embed)
        z = sample_z(mu, log_var)
        c = sample_c_prior(self.cf.c_dim, size=z.shape[0])
        y = self.decoder(word_seq, z, c)
        return {'pred': y, 'mu': mu, 'log_var': log_var}

    @staticmethod
    def loss(pred, mu, log_var, dec_target):
        n_vocab = pred.shape[-1]
        dec_target.t_()
        recon_loss = F.cross_entropy(
            pred.view(-1, n_vocab), dec_target.reshape(-1), reduction='mean'
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1 - log_var, 1))

        VariationalAE.update_hyper_params()
        return recon_loss + VariationalAE.kld_weight * kl_loss

    # word-around
    @staticmethod
    def init_hyper_params(n_iter, it_begin, kld_start_inc=3000, kld_weight=0.01, kld_max=0.15):
        VariationalAE.kld_start_inc = kld_start_inc
        VariationalAE.kld_weight = kld_weight
        VariationalAE.kld_max = kld_max
        VariationalAE.n_iter = n_iter
        VariationalAE.kld_inc = \
            (VariationalAE.kld_max - VariationalAE.kld_weight) / (n_iter - VariationalAE.kld_start_inc)
        VariationalAE.it = it_begin

    # word-around
    @staticmethod
    def update_hyper_params():
        # Anneal kl_weight
        VariationalAE.it += 1
        if VariationalAE.it > VariationalAE.kld_start_inc and \
                VariationalAE.kld_weight < VariationalAE.kld_max:
            VariationalAE.kld_weight += VariationalAE.kld_inc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
from .fast_config import FastConfig


class AutoEncoder(nn.Module):

    def __init__(self, cf: FastConfig):
        super(AutoEncoder, self).__init__()

        self.word_emb = WordEmbedding(cf)
        self.encoder = Encoder(cf)
        self.decoder = Decoder(self.word_emb, cf)
        self.discriminator = Discriminator(cf)
        self.cf = cf

        """
        Grouping the model's parameters
        """
        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder.parameters(), self.decoder.parameters()
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.auto_encoder_params)
        self.disc_params = filter(lambda p: p.requires_grad, self.discriminator.parameters())

        """
        Use GPU if set
        """
        if self.cf.gpu:
            self.cuda()

    def forward(self, *input):
        pass

    def compute_loss_vae(self, inputs, use_c_prior=True):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        self.train()
        batch_size = inputs.shape[0]
        pad_words = Variable(torch.LongTensor([self.cf.PAD_IDX])).repeat(1, batch_size)
        pad_words = pad_words.cuda() if self.cf.gpu else pad_words

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>' -> remove the <start> tag and add <pad> at the end
        sentence = torch.t(inputs)
        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        emb_inputs = self.word_emb(enc_inputs)
        mu, logvar = self.encoder(emb_inputs)
        z = self.sample_z(mu, logvar)

        if use_c_prior:
            c = self.sample_c_prior(batch_size)
        else:
            c = self.discriminator(sentence.transpose(0, 1))

        # Decoder: sentence -> y
        y = self.decoder(dec_inputs, z, c)

        recon_loss = F.cross_entropy(
            y.view(-1, self.cf.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1))

        return recon_loss, kl_loss

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.cf.z_dim))
        eps = eps.cuda() if self.cf.gpu else eps
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, size=1):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(size, self.cf.z_dim))
        z = z.cuda() if self.cf.gpu else z
        return z

    def sample_c_prior(self, size=1):
        """
        Sample c ~ p(c) = Cat([0.5, 0.5])
        """
        c = Variable(
            torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], size).astype('float32'))
        )
        c = c.cuda() if self.cf.gpu else c
        return c

    def generate_sentences(self, batch_size=32):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []
        cs = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)
            samples.append(self.sample_sentence(z, c, raw=True))
            cs.append(c.long())

        x_gen = torch.cat(samples, dim=0)
        c_gen = torch.cat(cs, dim=0)

        return x_gen, c_gen

    def sample_sentence(self, z, c, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.cf.START_IDX])
        word = word.cuda() if self.cf.gpu else word
        word = Variable(word)  # '<start>'

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        if raw:
            outputs.append(self.cf.START_IDX)

        for i in range(self.cf.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z, c], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output).view(-1)
            y = F.softmax(y / temp, dim=0)

            idx = torch.multinomial(y, 1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
            return outputs

    def generate_soft_embed(self, temp=1):
        """
        Generate soft embeddings of (batch_size x emb_dim) along with target z
        and c for each row (batch_size x {z_dim, c_dim})
        """
        samples = []
        targets_c = []
        targets_z = []

        for _ in range(self.batch_size):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)

            samples.append(self.sample_soft_embed(z, c, temp=1))
            targets_z.append(z)
            targets_c.append(c)

        x_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)
        _, targets_c = torch.cat(targets_c, dim=0).max(dim=1)

        return x_gen, targets_z, targets_c

    def sample_soft_embed(self, z, c, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        self.eval()

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.word_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z, c], 2)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = [self.word_emb(word).view(1, -1)]

        for _ in range(self.MAX_SENT_LEN):
            output, h = self.decoder(emb, h)
            o = self.decoder_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z, c], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs


class WordEmbedding(nn.Module):

    def __init__(self, cf: FastConfig):
        """
            Word embeddings layer
        """
        super(WordEmbedding, self).__init__()
        if cf.pre_trained_embeddings is None:
            self.emb_dim = cf.h_dim
            self.word_embedding = nn.Embedding(cf.n_vocab, self.emb_dim, self.PAD_IDX)
        else:
            self.emb_dim = cf.pre_trained_embeddings.size(1)
            self.word_embedding = nn.Embedding(cf.n_vocab, self.emb_dim, self.PAD_IDX)
            self.word_embedding.weight.data.copy_(cf.pre_trained_embeddings)        # Set pre-trained embeddings
            if cf.freeze_embeddings:
                self.word_embedding.weight.requires_grad = False

        # set embedding dimension on configuration
        cf.emb_dim = self.emb_dim

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        return self.word_embedding(inputs)


class Encoder(nn.Module):

    def __init__(self, cf: FastConfig):
        """
            Encoder is GRU with FC layers connected to last hidden unit
        """
        super(Encoder, self).__init__()
        self.encoder = nn.GRU(cf.emb_dim, cf.h_dim)
        self.q_mu = nn.Linear(cf.h_dim, cf.z_dim)           # mean
        self.q_log_var = nn.Linear(cf.h_dim, cf.z_dim)      # variance

    def forward(self, inputs):
        """
            Inputs is embeddings of: seq_len x batch_size x emb_dim
        """
        _, h = self.encoder(inputs, None)                   # h is the last output of GRU

        # Forward to latent
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        log_var = self.q_logvar(h)

        return mu, log_var


class Decoder(nn.Module):

    def __int__(self, word_emb: WordEmbedding, cf: FastConfig):
        """
            Decoder is GRU with `z` and `c` appended at its inputs
        """
        super(Decoder, self).__init__()
        self.word_emb = word_emb
        self.decoder = nn.GRU(cf.emb_dim + cf.z_dim + cf.c_dim, cf.z_dim + cf.c_dim, dropout=cf.word_dropout_prob)
        self.decoder_fc = nn.Linear(cf.z_dim + cf.c_dim, cf.n_vocab)

        self.n_vocab = cf.n_vocab
        self.word_dropout_prob = cf.word_dropout_prob
        self.UNK_IDX = cf.UNK_IDX
        self.gpu = cf.gpu

        if self.gpu:
            self.cuda()

    def forward(self, inputs, z, c):
        """
            Inputs must be embeddings: seq_len x batch_size
            z: unstructured latent code
            c: structured latent code
        """

        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x batch_size x (z_dim+c_dim)
        init_h = torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x batch_size x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, batch_size, _ = outputs.size()

        outputs = outputs.view(seq_len * batch_size, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, batch_size, self.n_vocab)

        return y

    def word_dropout(self, inputs):
        """
            Do word dropout: with prob `dropout`, set the word to '<unk>'.
        """

        data = inputs.data.clone() if isinstance(inputs, Variable) else inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(np.random.binomial(1, p=self.word_dropout_prob, size=tuple(data.size())).astype('uint8'))
        mask = mask.cuda() if self.gpu else mask

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)


class Discriminator(nn.Module):

    def __init__(self, cf: FastConfig):
        """
        Discriminator is CNN as in Kim, 2014
        """
        super(Discriminator, self).__init__()
        self.conv3 = nn.Conv2d(1, 100, (3, cf.emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, cf.emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, cf.emb_dim))
        self.disc_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, cf.c_dim)
        )

    def forward(self, inputs):
        """
            Inputs must be embeddings: batch_size x seq_len x emb_dim
        """
        inputs = inputs.unsqueeze(1)  # batch_size x 1 x seq_len x emb_dim

        x3 = F.relu(self.conv3(inputs)).squeeze()
        x4 = F.relu(self.conv4(inputs)).squeeze()
        x5 = F.relu(self.conv5(inputs)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.disc_fc(x)

        return y
        


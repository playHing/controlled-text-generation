class FastConfig:

    def __init__(self, h_dim, z_dim, c_dim, n_vocab,
                 unk_idx, pad_idx, start_idx, eos_idx,
                 max_sent_len=15, word_dropout_prob=0.3,
                 pre_trained_embeddings=None, freeze_embeddings=False, gpu=False):
        self.h_dim = h_dim
        self.emb_dim = "to be set by WordEmbedding"
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.n_vocab = n_vocab

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.word_dropout_prob = word_dropout_prob
        self.pre_trained_embeddings = pre_trained_embeddings
        self.freeze_embeddings = freeze_embeddings

        self.gpu = gpu

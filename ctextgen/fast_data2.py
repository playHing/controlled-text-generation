import os
import io

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler
from torchtext.vocab import Vocab,GloVe


class FastData2:

    def __init__(self, path=".data/sst2/", dataset="sst2", batch_size=32):
        if dataset == "sst2":
            dataset = DataSet()

            for db_set in ['full']:
                text_file = os.path.join(path, 'sst2-'+ db_set + '.text')
                label_file = os.path.join(path, 'sst2-'+ db_set + '.labels')
                with io.open(text_file, 'r', encoding="utf-8") as tf, io.open(label_file, 'r', encoding="utf-8") as lf:
                    for text in tf:
                        label = lf.readline()
                        dataset.append(Instance(text=text, label=label))
 
            dataset.apply(lambda x: x['text'].lower(), new_field_name='text')
            dataset.apply(lambda x: ['<start>'] + x['text'].split() + ['<eos>'], new_field_name='words')
            dataset.drop(lambda x: len(x['words']) > 17)
            dataset.apply(lambda x: x['words'] + ['<pad>'] * (17 - len(x['words'])), new_field_name='words')
            dataset.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

            # 不能随机分
            # _train_data, _test_data = dataset.split(0.3)
            _train_data = dataset

            dataset2 = DataSet()

            for db_set in ['test']:
                text_file = os.path.join(path, 'sst2-'+ db_set + '.text')
                label_file = os.path.join(path, 'sst2-'+ db_set + '.labels')
                with io.open(text_file, 'r', encoding="utf-8") as tf, io.open(label_file, 'r', encoding="utf-8") as lf:
                    for text in tf:
                        label = lf.readline()
                        dataset2.append(Instance(text=text, label=label))
 
            dataset2.apply(lambda x: x['text'].lower(), new_field_name='text')
            dataset2.apply(lambda x: ['<start>'] + x['text'].split() + ['<eos>'], new_field_name='words')
            dataset2.drop(lambda x: len(x['words']) > 17)
            dataset2.apply(lambda x: x['words'] + ['<pad>'] * (17 - len(x['words'])), new_field_name='words')
            dataset2.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

            _test_data = dataset2

            # _vocab = Vocabulary(min_freq=2)
            # TODO:设置min_freq会使得counter较大，影响vector
            _vocab = Vocabulary()
            _train_data.apply(lambda x: [_vocab.add(word) for word in x['words']])
            
            _vocab.build_vocab()
            del _vocab.word_count['<unk>']
            del _vocab.word_count['<pad>']
            del _vocab.word_count['<start>']
            del _vocab.word_count['<eos>']

            _train_data.apply(lambda x: [_vocab.to_index(word) for word in x['words']], new_field_name='word_seq',
                              is_input=True)
            _test_data.apply(lambda x: [_vocab.to_index(word) for word in x['words']], new_field_name='word_seq',
                             is_input=True)

        self.train_data = _train_data
        self.test_data = _test_data
        self.vocab = _vocab
        self.n_vocab = len(_vocab.idx2word)-3
        self.batch_size = batch_size
        self.train_iter = iter(Batch(dataset=self.train_data, batch_size=self.batch_size, sampler=SequentialSampler()))

    def next_batch(self):
        try:
            _next_batch = next(self.train_iter)
            if _next_batch[0]['word_seq'].shape[0] != self.batch_size:
                raise StopIteration
            return _next_batch
        except StopIteration:
            self.train_iter = iter(
                Batch(dataset=self.train_data, batch_size=self.batch_size, sampler=SequentialSampler()))
            return self.next_batch()
    
    def get_vocab_vectors(self):
        a = Vocab(self.vocab.word_count)
        a.load_vectors(vectors=GloVe('6B', dim=50))
        return a.vectors
    
    def idxs2sentence(self, idxs):
        return ' '.join([self.vocab.idx2word(i) for i in idxs])
    
    def tensor2sentence(self, idtensors):
        return ' '.join([self.vocab.to_word(int(i)) for i in idtensors])
    
    def idx2label(self, idx):
        return self.train_data['label_seq'][idx]

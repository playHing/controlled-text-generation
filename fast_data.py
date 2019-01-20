import os
import io

from torchtext import datasets, data

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler


class FastData:

    def __init__(self, path='.data/sst/trees', data_type='sst', test=False,
                 batch_size=32, split_ratio=0.1, seq_len=15, min_freq=2):

        data_set = DataSet()
        _test_data = DataSet()
        if data_type == 'yelp':
            path = '.data/yelp'
            for db_set in ['train']:
                text_file = os.path.join(path, 'sentiment.' + db_set + '.text')
                label_file = os.path.join(path, 'sentiment.' + db_set + '.labels')

                with io.open(text_file, 'r', encoding="utf-8") as tf, io.open(label_file, 'r', encoding="utf-8") as lf:
                    for text in tf:
                        label = lf.readline()
                        data_set.append(Instance(text=text, label=label))

            data_set.apply(lambda x: ['<start>'] + x['text'].lower().split() + ['<eos>'], new_field_name='words')
            data_set.drop(lambda x: len(x['words']) > seq_len + 2)

        elif data_type == 'sst':
            path = '.data/sst/trees'
            text = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
            label = data.Field(sequential=False, unk_token='<unk>')
            filter = lambda ex: len(ex.text) <= seq_len and ex.label != 'neutral'
            sst_train = datasets.SST(os.path.join(path, 'train.txt'), text, label, filter_pred=filter)
            sst_dev = datasets.SST(os.path.join(path, 'dev.txt'), text, label, filter_pred=filter)
            sst_test = datasets.SST(os.path.join(path, 'test.txt'), text, label, filter_pred=filter)
            for ex in sst_train.examples + sst_dev.examples + sst_test.examples:
                data_set.append(Instance(words=ex.text, label={'negative': 0, 'positive': 1}[ex.label]))

            data_set.apply(lambda x: ['<start>'] + [w.lower() for w in x['words']] + ['<eos>'], new_field_name='words')

        data_set.apply(lambda x: x['words'] + ['<pad>'] * (seq_len + 2 - len(x['words'])), new_field_name='words')

        _train_data = data_set
        print('# of sentences used to build vocab: {}'.format(len(_train_data)))

        _vocab = Vocabulary(min_freq=min_freq)
        _train_data.apply(lambda x: [_vocab.add(word) for word in x['words']])
        _vocab.build_vocab()

        if test:
            print('test mode.')
            #s = 0
            with io.open('fast_gen_data2.pos', 'r', encoding="utf-8") as f:
                for text in f:
                    _test_data.append(Instance(text=text, label=1))
            #        s += 1
            #        if s > 32 * 5:
            #            break
            #s = 0
            with io.open('fast_gen_data2.neg', 'r', encoding="utf-8") as f:
                for text in f:
                    _test_data.append(Instance(text=text, label=0))
            #        s += 1
            #        if s > 32 * 5:
            #            break

            _test_data.apply(lambda x: ['<start>'] + x['text'].lower().split() + ['<eos>'], new_field_name='words')
            _test_data.drop(lambda x: len(x['words']) > seq_len + 2)
            _test_data.apply(lambda x: x['words'] + ['<pad>'] * (seq_len + 2 - len(x['words'])), new_field_name='words')
            data_set = _test_data

        data_set.apply(lambda x: [_vocab.to_index(w) for w in x['words']], new_field_name='word_seq', is_input=True)
        data_set.apply(lambda x: x['word_seq'][1:] + [0], new_field_name='dec_target', is_target=True)
        data_set.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)
        _train_data, _test_data = data_set.split(split_ratio)

        self.train_data = _train_data
        self.test_data = _test_data
        self.vocab = _vocab
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

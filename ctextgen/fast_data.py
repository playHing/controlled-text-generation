import os
import io

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler


class FastData:

    def __init__(self, path=".data/yelp", data_type="yelp",
                 batch_size=32, split_ratio=0.1, min_freq=2):

        if data_type == "yelp":
            data_set = DataSet()

            for db_set in ['train']:
                text_file = os.path.join(path, 'sentiment.' + db_set + '.text')
                label_file = os.path.join(path, 'sentiment.' + db_set + '.labels')
                TEST = 0
                with io.open(text_file, 'r', encoding="utf-8") as tf, io.open(label_file, 'r', encoding="utf-8") as lf:
                    for text in tf:
                        label = lf.readline()
                        data_set.append(Instance(text=text, label=label))
                        if TEST > 1000:
                            break
                        else:
                            TEST += 1

        data_set.apply(lambda x: ['<start>'] + x['text'].lower().split() + ['<eos>'], new_field_name='words')
        data_set.drop(lambda x: len(x['words']) > 17)
        data_set.apply(lambda x: x['words'] + ['<pad>'] * (17 - len(x['words'])), new_field_name='words')

        _train_data, _ = data_set.split(split_ratio)

        _vocab = Vocabulary(min_freq=min_freq)
        _train_data.apply(lambda x: [_vocab.add(word) for word in x['words']])
        _vocab.build_vocab()

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

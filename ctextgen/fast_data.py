import os
import io

from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler


class FastData:

    def __init__(self, path=".data/yelp", dataset="yelp", batch_size=32):

        if dataset == "yelp":
            dataset = DataSet()

            for db_set in ['train']:
                text_file = os.path.join(path, 'sentiment.' + db_set + '.text')
                label_file = os.path.join(path, 'sentiment.' + db_set + '.labels')
                with io.open(text_file, 'r', encoding="utf-8") as tf, io.open(label_file, 'r', encoding="utf-8") as lf:
                    for text in tf:
                        label = lf.readline()
                        dataset.append(Instance(text=text, label=label))

            dataset.apply(lambda x: x['text'].lower(), new_field_name='text')
            dataset.apply(lambda x: ['<start>'] + x['text'].split() + ['<eos>'], new_field_name='words')
            dataset.drop(lambda x: len(x['words']) > 17)
            dataset.apply(lambda x: x['words'] + ['<pad>'] * (17 - len(x['words'])), new_field_name='words')
            dataset.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

            _train_data, _test_data = dataset.split(0.3)

            _vocab = Vocabulary(min_freq=2)
            _train_data.apply(lambda x: [_vocab.add(word) for word in x['words']])
            _vocab.build_vocab()

            _train_data.apply(lambda x: [_vocab.to_index(word) for word in x['words']], new_field_name='word_seq',
                              is_input=True)
            _test_data.apply(lambda x: [_vocab.to_index(word) for word in x['words']], new_field_name='word_seq',
                             is_input=True)

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

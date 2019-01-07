import glob
import os
import io

from fastNLP import DataSet, Instance, Vocabulary


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
        dataset.apply(lambda x: x['text'].split(), new_field_name='words')
        dataset.drop(lambda x: len(x['words']) > 15)

        _test_data, _train_data = dataset.split(0.3)

        vocab = Vocabulary(min_freq=2)
        _train_data.apply(lambda x: [vocab.add(word) for word in x['words']])
        vocab.build_vocab()

        _train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
        _test_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)

        return _train_data, _test_data, vocab


train_data, test_data, vocab = prepare_data(".data/yelp")



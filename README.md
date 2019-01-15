# Ctrl-gen
Reproducing Hu, et. al., ICML 2017's "Toward Controlled Generation of Text" in PyTorch with FastNLP.
This work is forked from University of Bonn's NLP Lab project on Winter Semester 2017/2018,
and for DL course in FDU.


## Requirements
1. Python 3.6+
2. PyTorch 1.0+
3. fastNLP 0.2 <https://github.com/fastnlp/fastNLP>
4. TorchText <https://github.com/pytorch/text>


## How to run
0. Download `yelp` and `SST` and place them in folder `.data`
1. Run `python train_vae.py --save --train_vae {--gpu}`. This will create `vae.bin`. Essentially this is the base VAE as in Bowman, 2015 [2]. This will also create `disc.bin`. The discriminator is using Kim, 2014 [3] architecture and the training procedure is as in Hu, 2017 [1].

~~2. Run `test.py --model {vae, ctextgen}.bin {--gpu}` for basic evaluations, e.g. conditional generation and latent interpolation.~~

## Difference compared to the paper
1. Only conditions the model with sentiment, i.e. no tense conditioning.
2. Entirely using SST dataset, which has only ~2800 sentences after filtering. This might not be enough and leads to overfitting. The base VAE in the original model by Hu, 2017 [1] is trained using larger dataset first.
3. Obviously most of the hyperparameters values are different.


## Dataset

SST <http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip>

yelp <https://drive.google.com/file/d/1HaUKEYDBEk6GlJGmXwqYteB-4rS9q8Lg/view?usp=sharing>

~~IMDB~~<http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>

## References
1. Hu, Zhiting, et al. "Toward controlled generation of text." International Conference on Machine Learning. 2017. [[pdf](http://proceedings.mlr.press/v70/hu17e/hu17e.pdf)]
2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015). [[pdf](https://arxiv.org/pdf/1511.06349.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)]
3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014). [[pdf](https://arxiv.org/pdf/1408.5882)]
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Shiva Manne <manneshiva@gmail.com>, Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Learn word representations via Fasttext: `Enriching Word Vectors with Subword Information
<https://arxiv.org/abs/1607.04606>`_.

This module allows training word embeddings from a training corpus with the additional ability to obtain word vectors
for out-of-vocabulary words.

This module contains a fast native C implementation of Fasttext with Python interfaces. It is **not** only a wrapper
around Facebook's implementation.

For a tutorial see `this notebook
<https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb>`_.

**Make sure you have a C compiler before installing Gensim, to use the optimized (compiled) Fasttext
training routines.**

Usage examples
--------------

Initialize and train a model:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.models import FastText
    >>>
    >>> model = FastText(common_texts, size=4, window=3, min_count=1, iter=10)

Persist a model to disk with:

.. sourcecode:: pycon

    >>> from gensim.test.utils import get_tmpfile
    >>>
    >>> fname = get_tmpfile("fasttext.model")
    >>>
    >>> model.save(fname)
    >>> model = FastText.load(fname)  # you can continue training with the loaded model!

Retrieve word-vector for vocab and out-of-vocab word:

.. sourcecode:: pycon

    >>> existent_word = "computer"
    >>> existent_word in model.wv.vocab
    True
    >>> computer_vec = model.wv[existent_word]  # numpy vector of a word
    >>>
    >>> oov_word = "graph-out-of-vocab"
    >>> oov_word in model.wv.vocab
    False
    >>> oov_vec = model.wv[oov_word]  # numpy vector for OOV word

You can perform various NLP word tasks with the model, some of them are already built-in:

.. sourcecode:: pycon

    >>> similarities = model.wv.most_similar(positive=['computer', 'human'], negative=['interface'])
    >>> most_similar = similarities[0]
    >>>
    >>> similarities = model.wv.most_similar_cosmul(positive=['computer', 'human'], negative=['interface'])
    >>> most_similar = similarities[0]
    >>>
    >>> not_matching = model.wv.doesnt_match("human computer interface tree".split())
    >>>
    >>> sim_score = model.wv.similarity('computer', 'human')

Correlation with human opinion on word similarity:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies:

.. sourcecode:: pycon

    >>> analogies_result = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

"""

import logging

import numpy as np
from numpy import ones, vstack, float32 as REAL, sum as np_sum
import six

import gensim.models._fasttext_bin

from gensim.models.word2vec import Word2VecVocab, Word2VecTrainables, train_sg_pair, train_cbow_pair
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.utils_any2vec import _compute_ngrams, _ft_hash, _ft_hash_broken
from smart_open import smart_open

from gensim.utils import deprecated, call_on_class_only

logger = logging.getLogger(__name__)

try:
    from gensim.models.fasttext_inner import train_batch_sg, train_batch_cbow
    from gensim.models.fasttext_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

    def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
        """Update CBOW model by training on a sequence of sentences.

        Called internally from :meth:`~gensim.models.fasttext.FastText.train`.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim will use the optimized version
        from :mod:`gensim.models.fasttext_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.fasttext.FastText`
            Model instance.
        sentences : iterable of list of str
            Iterable of the sentences.
        alpha : float
            Learning rate.
        work : :class:`numpy.ndarray`, optional
            UNUSED.
        neu1 : :class:`numpy.ndarray`, optional
            UNUSED.
        Returns
        -------
        int
            Effective number of words trained.

        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab
                           and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

                vocab_subwords_indices = []
                ngrams_subwords_indices = []

                for index in word2_indices:
                    vocab_subwords_indices += [index]
                    ngrams_subwords_indices.extend(model.wv.buckets_word[index])

                l1_vocab = np_sum(model.wv.syn0_vocab[vocab_subwords_indices], axis=0)  # 1 x vector_size
                l1_ngrams = np_sum(model.wv.syn0_ngrams[ngrams_subwords_indices], axis=0)  # 1 x vector_size

                l1 = np_sum([l1_vocab, l1_ngrams], axis=0)
                subwords_indices = [vocab_subwords_indices] + [ngrams_subwords_indices]
                if (subwords_indices[0] or subwords_indices[1]) and model.cbow_mean:
                    l1 /= (len(subwords_indices[0]) + len(subwords_indices[1]))

                # train on the sliding window for target word
                train_cbow_pair(model, word, subwords_indices, l1, alpha, is_ft=True)
            result += len(word_vocabs)
        return result

    def train_batch_sg(model, sentences, alpha, work=None, neu1=None):
        """Update skip-gram model by training on a sequence of sentences.

        Called internally from :meth:`~gensim.models.fasttext.FastText.train`.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim will use the optimized version
        from :mod:`gensim.models.fasttext_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.fasttext.FastText`
            `FastText` instance.
        sentences : iterable of list of str
            Iterable of the sentences directly from disk/network.
        alpha : float
            Learning rate.
        work : :class:`numpy.ndarray`, optional
            UNUSED.
        neu1 : :class:`numpy.ndarray`, optional
            UNUSED.

        Returns
        -------
        int
            Effective number of words trained.

        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab
                           and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)

                subwords_indices = (word.index,)
                subwords_indices += tuple(model.wv.buckets_word[word.index])

                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    if pos2 != pos:  # don't train on the `word` itself
                        train_sg_pair(model, model.wv.index2word[word2.index], subwords_indices, alpha, is_ft=True)

            result += len(word_vocabs)
        return result

try:
    from gensim.models.fasttext_corpusfile import train_epoch_sg, train_epoch_cbow, CORPUSFILE_VERSION
except ImportError:
    # file-based fasttext is not supported
    CORPUSFILE_VERSION = -1

    def train_epoch_sg(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
                       _work, _neu1):
        raise RuntimeError("Training with corpus_file argument is not supported")

    def train_epoch_cbow(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
                         _work, _neu1):
        raise RuntimeError("Training with corpus_file argument is not supported")


FASTTEXT_FILEFORMAT_MAGIC = 793712314


class FastText(BaseWordEmbeddingsModel):
    """Train, use and evaluate word representations learned using the method
    described in `Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_, aka FastText.

    The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save` and
    :meth:`~gensim.models.fasttext.FastText.load` methods, or loaded from a format compatible with the original
    Fasttext implementation via :meth:`~gensim.models.fasttext.FastText.load_fasttext_format`.

    Some important internal attributes are the following:

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.FastTextKeyedVectors`
        This object essentially contains the mapping between words and embeddings. These are similar to the embeddings
        computed in the :class:`~gensim.models.word2vec.Word2Vec`, however here we also include vectors for n-grams.
        This allows the model to compute embeddings even for **unseen** words (that do not exist in the vocabulary),
        as the aggregate of the n-grams included in the word. After training the model, this attribute can be used
        directly to query those embeddings in various ways. Check the module level docstring for some examples.
    vocabulary : :class:`~gensim.models.fasttext.FastTextVocab`
        This object represents the vocabulary of the model.
        Besides keeping track of all unique words, this object provides extra functionality, such as
        constructing a huffman tree (frequent words are closer to the root), or discarding extremely rare words.
    trainables : :class:`~gensim.models.fasttext.FastTextTrainables`
        This object represents the inner shallow neural network used to train the embeddings. This is very
        similar to the network of the :class:`~gensim.models.word2vec.Word2Vec` model, but it also trains weights
        for the N-Grams (sequences of more than 1 words). The semantics of the network are almost the same as
        the one used for the :class:`~gensim.models.word2vec.Word2Vec` model.
        You can think of it as a NN with a single projection and hidden layer which we train on the corpus.
        The weights are then used as our embeddings. An important difference however between the two models, is the
        scoring function used to compute the loss. In the case of FastText, this is modified in word to also account
        for the internal structure of words, besides their concurrence counts.

    """
    def __init__(self, sentences=None, corpus_file=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6,
                 sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(),
                 compatible_hash=True):
        """

        Parameters
        ----------
        sentences : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (or none of them).
        min_count : int, optional
            The model ignores all words with total frequency lower than this.
        size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : {1, 0}, optional
            Training algorithm: skip-gram if `sg=1`, otherwise CBOW.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupr??, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        cbow_mean : {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.fasttext.FastText.build_vocab` and is not stored as part of themodel.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        sorted_vocab : {1,0}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indices.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        min_n : int, optional
            Minimum length of char n-grams to be used for training word representations.
        max_n : int, optional
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        word_ngrams : {1,0}, optional
            If 1, uses enriches word vectors with subword(n-grams) information.
            If 0, this is equivalent to :class:`~gensim.models.word2vec.Word2Vec`.
        bucket : int, optional
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.

        compatible_hash: bool, optional
            By default, newer versions of Gensim's FastText use a hash function
            that is 100% compatible with Facebook's FastText.
            Older versions were not 100% compatible due to a bug.
            To use the older, incompatible hash function, set this to False.

        Examples
        --------
        Initialize and train a `FastText` model:

        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = FastText(sentences, min_count=1)
            >>> say_vector = model.wv['say']  # get vector for word
            >>> of_vector = model.wv['of']  # get vector for out-of-vocab word

        """
        self.load = call_on_class_only
        self.load_fasttext_format = call_on_class_only
        self.callbacks = callbacks
        self.word_ngrams = int(word_ngrams)
        if self.word_ngrams <= 1 and max_n == 0:
            bucket = 0

        self.wv = FastTextKeyedVectors(size, min_n, max_n, bucket, compatible_hash)
        self.vocabulary = FastTextVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=bool(sorted_vocab), null_word=null_word, ns_exponent=ns_exponent)
        self.trainables = FastTextTrainables(vector_size=size, seed=seed, bucket=bucket, hashfxn=hashfxn)
        self.trainables.prepare_weights(hs, negative, self.wv, update=False, vocabulary=self.vocabulary)
        self.wv.bucket = self.trainables.bucket

        super(FastText, self).__init__(
            sentences=sentences, corpus_file=corpus_file, workers=workers, vector_size=size, epochs=iter,
            callbacks=callbacks, batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window,
            seed=seed, hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, fast_version=FAST_VERSION)

    @property
    @deprecated("Attribute will be removed in 4.0.0, use wv.min_n instead")
    def min_n(self):
        return self.wv.min_n

    @property
    @deprecated("Attribute will be removed in 4.0.0, use wv.max_n instead")
    def max_n(self):
        return self.wv.max_n

    @property
    @deprecated("Attribute will be removed in 4.0.0, use trainables.bucket instead")
    def bucket(self):
        return self.trainables.bucket

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self):
        return self.trainables.vectors_vocab_lockf

    @syn0_vocab_lockf.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self, value):
        self.trainables.vectors_vocab_lockf = value

    @syn0_vocab_lockf.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self):
        del self.trainables.vectors_vocab_lockf

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self):
        return self.trainables.vectors_ngrams_lockf

    @syn0_ngrams_lockf.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self, value):
        self.trainables.vectors_ngrams_lockf = value

    @syn0_ngrams_lockf.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self):
        del self.trainables.vectors_ngrams_lockf

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.num_ngram_vectors instead")
    def num_ngram_vectors(self):
        return self.wv.num_ngram_vectors

    def build_vocab(self, sentences=None, corpus_file=None, update=False, progress_per=10000, keep_raw_vocab=False,
                    trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        Parameters
        ----------
        sentences : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (not both of them).
        update : bool
            If true, the new words in `sentences` will be added to model's vocab.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.fasttext.FastText.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        **kwargs
            Additional key word parameters passed to
            :meth:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel.build_vocab`.

        Examples
        --------
        Train a model and update vocab for online training:

        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>> sentences_2 = [["dude", "say", "wazzup!"]]
            >>>
            >>> model = FastText(min_count=1)
            >>> model.build_vocab(sentences_1)
            >>> model.train(sentences_1, total_examples=model.corpus_count, epochs=model.epochs)
            >>>
            >>> model.build_vocab(sentences_2, update=True)
            >>> model.train(sentences_2, total_examples=model.corpus_count, epochs=model.epochs)

        """
        if update:
            if not len(self.wv.vocab):
                raise RuntimeError(
                    "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                    "First build the vocabulary of your model with a corpus "
                    "before doing an online update.")
            self.vocabulary.old_vocab_len = len(self.wv.vocab)
            self.trainables.old_hash2index_len = len(self.wv.hash2index)

        return super(FastText, self).build_vocab(
            sentences=sentences, corpus_file=corpus_file, update=update, progress_per=progress_per,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)

    def _set_train_params(self, **kwargs):
        #
        # We need the wv.buckets_word member to be initialized in order to
        # continue training.  The _clear_post_train method destroys this
        # variable, so we reinitialize it here, if needed.
        #
        # The .old_vocab_len and .old_hash2index_len members are set only to
        # keep the init_ngrams_weights method happy.
        #
        if self.wv.buckets_word is None:
            self.vocabulary.old_vocab_len = len(self.wv.vocab)
            self.trainables.old_hash2index_len = len(self.wv.hash2index)
            self.trainables.init_ngrams_weights(self.wv, update=True, vocabulary=self.vocabulary)

    def _clear_post_train(self):
        """Clear the model's internal structures after training has finished to free up RAM."""
        self.wv.vectors_norm = None
        self.wv.vectors_vocab_norm = None
        self.wv.vectors_ngrams_norm = None
        self.wv.buckets_word = None

    def estimate_memory(self, vocab_size=None, report=None):
        hash_fn = _ft_hash if self.wv.compatible_hash else _ft_hash_broken

        vocab_size = vocab_size or len(self.wv.vocab)
        vec_size = self.vector_size * np.dtype(np.float32).itemsize
        l1_size = self.trainables.layer1_size * np.dtype(np.float32).itemsize
        report = report or {}
        report['vocab'] = len(self.wv.vocab) * (700 if self.hs else 500)
        report['syn0_vocab'] = len(self.wv.vocab) * vec_size
        num_buckets = self.trainables.bucket
        if self.hs:
            report['syn1'] = len(self.wv.vocab) * l1_size
        if self.negative:
            report['syn1neg'] = len(self.wv.vocab) * l1_size
        if self.word_ngrams > 0 and self.wv.vocab:
            num_buckets = num_ngrams = 0

            if self.trainables.bucket:
                buckets = set()
                num_ngrams = 0
                for word in self.wv.vocab:
                    ngrams = _compute_ngrams(word, self.wv.min_n, self.wv.max_n)
                    num_ngrams += len(ngrams)
                    buckets.update(hash_fn(ng) % self.trainables.bucket for ng in ngrams)
                num_buckets = len(buckets)
            report['syn0_ngrams'] = num_buckets * vec_size
            # A tuple (48 bytes) with num_ngrams_word ints (8 bytes) for each word
            # Only used during training, not stored with the model
            report['buckets_word'] = 48 * len(self.wv.vocab) + 8 * num_ngrams
        elif self.word_ngrams > 0:
            logger.warn(
                'subword information is enabled, but no vocabulary could be found, estimated required memory might be '
                'inaccurate!'
            )
        report['total'] = sum(report.values())
        logger.info(
            "estimated required memory for %i words, %i buckets and %i dimensions: %i bytes",
            len(self.wv.vocab), num_buckets, self.vector_size, report['total']
        )
        return report

    def _do_train_epoch(self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch,
                        total_examples=None, total_words=None, **kwargs):
        work, neu1 = thread_private_mem

        if self.sg:
            examples, tally, raw_tally = train_epoch_sg(self, corpus_file, offset, cython_vocab, cur_epoch,
                                                        total_examples, total_words, work, neu1)
        else:
            examples, tally, raw_tally = train_epoch_cbow(self, corpus_file, offset, cython_vocab, cur_epoch,
                                                          total_examples, total_words, work, neu1)

        return examples, tally, raw_tally

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.

        Parameters
        ----------
        sentences : iterable of list of str
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        alpha : float
            The current learning rate.
        inits : tuple of (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Each worker's private work memory.

        Returns
        -------
        (int, int)
            Tuple of (effective word count after ignoring unknown words and sentence length trimming, total word count)

        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work, neu1)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)

    def train(self, sentences=None, corpus_file=None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For FastText, each sentence must be a list of unicode strings.

        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate
        progress-percentage logging, either `total_examples` (count of sentences) or `total_words` (count of
        raw words in sentences) **MUST** be provided. If `sentences` is the same corpus
        that was provided to :meth:`~gensim.models.fasttext.FastText.build_vocab` earlier,
        you can simply use `total_examples=self.corpus_count`.

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case
        where :meth:`~gensim.models.fasttext.FastText.train` is only called once, you can set `epochs=self.iter`.

        Parameters
        ----------
        sentences : iterable of list of str, optional
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            If you use this argument instead of `sentences`, you must provide `total_words` argument as well. Only one
            of `sentences` or `corpus_file` arguments need to be passed (not both of them).
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to :meth:`~gensim.models.fasttext.FastText.train`.
            Use only if making multiple calls to :meth:`~gensim.models.fasttext.FastText.train`, when you want to manage
            the alpha learning-rate yourself (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to
            :meth:`~gensim.models.fasttext.FastText.train`.
            Use only if making multiple calls to :meth:`~gensim.models.fasttext.FastText.train`, when you want to manage
            the alpha learning-rate yourself (not recommended).
        word_count : int
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float
            Seconds to wait before reporting progress.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = FastText(min_count=1)
            >>> model.build_vocab(sentences)
            >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        """
        super(FastText, self).train(
            sentences=sentences, corpus_file=corpus_file, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks)
        self.wv.adjust_vectors()

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool
            If True, forget the original vectors and only keep the normalized ones to save RAM.

        """
        # init_sims() resides in KeyedVectors because it deals with input layer mainly, but because the
        # hidden layer is not an attribute of KeyedVectors, it has to be deleted in this class.
        # The normalizing of input layer happens inside of KeyedVectors.
        if replace and hasattr(self.trainables, 'syn1'):
            del self.trainables.syn1
        self.wv.init_sims(replace)

    def clear_sims(self):
        """Remove all L2-normalized word vectors from the model, to free up memory.

        You can recompute them later again using the :meth:`~gensim.models.fasttext.FastText.init_sims` method.

        """
        self._clear_post_train()

    @deprecated("Method will be removed in 4.0.0, use self.wv.__getitem__() instead")
    def __getitem__(self, words):
        """Deprecated. Use self.wv.__getitem__() instead.

        Refer to the documentation for :meth:`gensim.models.keyedvectors.KeyedVectors.__getitem__`

        """
        return self.wv.__getitem__(words)

    @deprecated("Method will be removed in 4.0.0, use self.wv.__contains__() instead")
    def __contains__(self, word):
        """Deprecated. Use self.wv.__contains__() instead.

        Refer to the documentation for :meth:`gensim.models.keyedvectors.KeyedVectors.__contains__`

        """
        return self.wv.__contains__(word)

    @classmethod
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """Load the input-hidden weight matrix from Facebook's native fasttext `.bin` and `.vec` output files.

        Notes
        ------
        Due to limitations in the FastText API, you cannot continue training with a model loaded this way.

        Parameters
        ----------
        model_file : str
            Path to the FastText output files.
            FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`
            Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
            as Gensim requires only `.bin` file to the load entire fastText model.
        encoding : str, optional
            Specifies the file encoding.

        Returns
        -------
        :class: `~gensim.models.fasttext.FastText`
            The loaded model.

        """
        return _load_fasttext_format(model_file, encoding=encoding)

    def load_binary_data(self, encoding='utf8'):
        """Load data from a binary file created by Facebook's native FastText.

        Parameters
        ----------
        encoding : str, optional
            Specifies the encoding.

        """
        m = _load_fasttext_format(self.file_name, encoding=encoding)
        for attr, val in six.iteritems(m.__dict__):
            setattr(self, attr, val)

    def save(self, *args, **kwargs):
        """Save the Fasttext model. This saved model can be loaded again using
        :meth:`~gensim.models.fasttext.FastText.load`, which supports incremental training
        and getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Store the model to this file.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastText.load`
            Load :class:`~gensim.models.fasttext.FastText` model.

        """
        kwargs['ignore'] = kwargs.get(
            'ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm', 'buckets_word'])
        super(FastText, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved `FastText` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~gensim.models.fasttext.FastText`
            Loaded model.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastText.save`
            Save :class:`~gensim.models.fasttext.FastText` model.

        """
        try:
            model = super(FastText, cls).load(*args, **kwargs)
            if not hasattr(model.trainables, 'vectors_vocab_lockf') and hasattr(model.wv, 'vectors_vocab'):
                model.trainables.vectors_vocab_lockf = ones(model.wv.vectors_vocab.shape, dtype=REAL)
            if not hasattr(model.trainables, 'vectors_ngrams_lockf') and hasattr(model.wv, 'vectors_ngrams'):
                model.trainables.vectors_ngrams_lockf = ones(model.wv.vectors_ngrams.shape, dtype=REAL)

            if not hasattr(model.wv, 'compatible_hash'):
                logger.warning(
                    "This older model was trained with a buggy hash function.  ",
                    "The model will continue to work, but consider training it "
                    "from scratch."
                )
                model.wv.compatible_hash = False

            if not hasattr(model.wv, 'bucket'):
                model.wv.bucket = model.trainables.bucket

            return model
        except AttributeError:
            logger.info('Model saved using code from earlier Gensim Version. Re-loading old model in a compatible way.')
            from gensim.models.deprecated.fasttext import load_old_fasttext
            return load_old_fasttext(*args, **kwargs)

    @deprecated("Method will be removed in 4.0.0, use self.wv.accuracy() instead")
    def accuracy(self, questions, restrict_vocab=30000, most_similar=None, case_insensitive=True):
        most_similar = most_similar or FastTextKeyedVectors.most_similar
        return self.wv.accuracy(questions, restrict_vocab, most_similar, case_insensitive)


#
# Keep for backward compatibility.
#
class FastTextVocab(Word2VecVocab):
    pass


class FastTextTrainables(Word2VecTrainables):
    """Represents the inner shallow neural network used to train :class:`~gensim.models.fasttext.FastText`."""
    def __init__(self, vector_size=100, seed=1, hashfxn=hash, bucket=2000000):
        super(FastTextTrainables, self).__init__(
            vector_size=vector_size, seed=seed, hashfxn=hashfxn)
        self.bucket = int(bucket)

        #
        # There are also two "hidden" attributes that get initialized outside
        # this constructor:
        #
        #   1. vectors_vocab_lockf
        #   2. vectors_ngrams_lockf
        #
        # These are both 2D matrices of shapes equal to the shapes of
        # wv.vectors_vocab and wv.vectors_ngrams.  So, each row corresponds to
        # a vector, and each column corresponds to a dimension within that
        # vector.
        #
        # Lockf stands for "lock factor": zero values suppress learning, one
        # values enable it.  Interestingly, the vectors_vocab_lockf and
        # vectors_ngrams_lockf seem to be used only by the C code in
        # fasttext_inner.pyx.
        #
        # The word2vec implementation also uses vectors_lockf: in that case,
        # it's a 1D array, with a real number for each vector.  The FastText
        # implementation inherits this vectors_lockf attribute but doesn't
        # appear to use it.
        #

    def prepare_weights(self, hs, negative, wv, update=False, vocabulary=None):
        super(FastTextTrainables, self).prepare_weights(hs, negative, wv, update=update, vocabulary=vocabulary)
        self.init_ngrams_weights(wv, update=update, vocabulary=vocabulary)

    def init_ngrams_weights(self, wv, update=False, vocabulary=None):
        """Compute ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.

        Parameters
        ----------
        wv : :class:`~gensim.models.keyedvectors.FastTextKeyedVectors`
            Contains the mapping between the words and embeddings.
            The vectors for the computed ngrams will go here.
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.
        vocabulary : :class:`~gensim.models.fasttext.FastTextVocab`
            This object represents the vocabulary of the model.
            If update is True, then vocabulary may not be None.

        """
        if not update:
            wv.init_ngrams_weights(self.seed)
            self.vectors_vocab_lockf = ones(wv.vectors_vocab.shape, dtype=REAL)
            self.vectors_ngrams_lockf = ones(wv.vectors_ngrams.shape, dtype=REAL)
        else:
            wv.update_ngrams_weights(self.seed, vocabulary.old_vocab_len)
            self.vectors_vocab_lockf = _pad_ones(self.vectors_vocab_lockf, wv.vectors_vocab.shape)
            self.vectors_ngrams_lockf = _pad_ones(self.vectors_ngrams_lockf, wv.vectors_ngrams.shape)

    def init_post_load(self, model, hidden_output):
        num_vectors = len(model.wv.vectors)
        vocab_size = len(model.wv.vocab)
        vector_size = model.wv.vector_size

        assert num_vectors > 0, 'expected num_vectors to be initialized already'
        assert vocab_size > 0, 'expected vocab_size to be initialized already'

        self.vectors_ngrams_lockf = ones(model.wv.vectors_ngrams.shape, dtype=REAL)
        self.vectors_vocab_lockf = ones(model.wv.vectors_vocab.shape, dtype=REAL)

        if model.hs:
            self.syn1 = hidden_output
        if model.negative:
            self.syn1neg = hidden_output

        self.layer1_size = vector_size


def _pad_ones(m, new_shape):
    """Pad a matrix with additional rows filled with ones."""
    assert m.shape[0] <= new_shape[0], 'the new number of rows must be greater'
    assert m.shape[1] == new_shape[1], 'the number of columns must match'
    new_rows = new_shape[0] - m.shape[0]
    if new_rows == 0:
        return m
    suffix = ones((new_rows, m.shape[1]), dtype=REAL)
    return vstack([m, suffix])


def _load_fasttext_format(model_file, encoding='utf-8'):
    """Load the input-hidden weight matrix from Facebook's native fasttext `.bin` and `.vec` output files.

    Parameters
    ----------
    model_file : str
        Path to the FastText output files.
        FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`
        Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
        as Gensim requires only `.bin` file to the load entire fastText model.
    encoding : str, optional
        Specifies the file encoding.

    Returns
    -------
    :class: `~gensim.models.fasttext.FastText`
        The loaded model.
    """
    if not model_file.endswith('.bin'):
        model_file += '.bin'
    with smart_open(model_file, 'rb') as fin:
        m = gensim.models._fasttext_bin.load(fin, encoding=encoding)

    model = FastText(
        size=m.dim,
        window=m.ws,
        iter=m.epoch,
        negative=m.neg,
        hs=(m.loss == 1),
        sg=(m.model == 2),
        bucket=m.bucket,
        min_count=m.min_count,
        sample=m.t,
        min_n=m.minn,
        max_n=m.maxn,
    )

    model.vocabulary.raw_vocab = m.raw_vocab
    model.vocabulary.nwords = m.nwords
    model.vocabulary.vocab_size = m.vocab_size
    model.vocabulary.prepare_vocab(model.hs, model.negative, model.wv,
                                   update=True, min_count=model.min_count)

    model.num_original_vectors = m.vectors_ngrams.shape[0]

    model.wv.init_post_load(m.vectors_ngrams)
    model.trainables.init_post_load(model, m.hidden_output)

    _check_model(model)

    logger.info("loaded %s weight matrix for fastText model from %s", m.vectors_ngrams.shape, fin.name)
    return model


def _check_model(m):
    #
    # These checks only make sense after everything has been completely initialized.
    #
    assert m.wv.vector_size == m.wv.vectors_ngrams.shape[1], (
        'mismatch between vector size in model params ({}) and model vectors ({})'
        .format(m.wv.vector_size, m.wv.vectors_ngrams)
    )
    if m.trainables.syn1neg is not None:
        assert m.wv.vector_size == m.trainables.syn1neg.shape[1], (
            'mismatch between vector size in model params ({}) and trainables ({})'
            .format(m.wv.vector_size, m.wv.vectors_ngrams)
        )

    assert len(m.wv.vocab) == m.vocabulary.nwords, (
        'mismatch between final vocab size ({} words), '
        'and expected number of words ({} words)'.format(len(m.wv.vocab), m.vocabulary.nwords)
    )

    if len(m.wv.vocab) != m.vocabulary.vocab_size:
        # expecting to log this warning only for pretrained french vector, wiki.fr
        logger.warning(
            "mismatch between final vocab size (%s words), and expected vocab size (%s words)",
            len(m.wv.vocab), m.vocabulary.vocab_size
        )

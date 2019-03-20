from models import Skipgram
from utils import WordSegmenter
import tensorflow as tf
from collections import namedtuple
from models.Fasttext import denseNDArrayToSparseTensor
import numpy as np
from copy import copy


class SubwordCNN(Skipgram):
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 segmenter_path=None,
                 max_segments=None,
                 negative=None,
                 n_context=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None,
                 model_name='subwordcnn',
                 batch_size=None):
        self.graph_path = graph_path
        self.vocabulary_size = vocab_size
        self.emb_size = emb_size
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.gram_segmenter = WordSegmenter(segmenter_path[0], max_segments[0], vocab_size, include_word=False)
        self.morph_segmenter = WordSegmenter(segmenter_path[1], max_segments[1], vocab_size, include_word=False)
        self.lemma_segmenter = WordSegmenter(segmenter_path[2], max_segments[2], vocab_size, include_word=False)

        self.max_grams = self.gram_segmenter.max_len
        self.max_morph = self.morph_segmenter.max_len
        self.segm_voc_size = self.morph_segmenter.unique_segments + \
                             self.lemma_segmenter.unique_segments + \
                             self.gram_segmenter.unique_segments

        self.h = {
            'feat_emb_size': 100,
            'd_hid': 500,
            'd_out': emb_size,
        }

        self.t_words = n_context + 1 + negative
        self.batch_size = batch_size
        self.n_neg = negative
        self.context_size = n_context
        self.temp = 100.
        # self.feat_space = self.gram_segmenter.unique_segments + self.morph_segmenter.unique_segments + self.lemma_segmenter.unique_segments

        self.assemble_model()

        self.open_session(gpu_options)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.graph_path,
                                                    graph=self.sess.graph)

    # def assemble_model_orig(self):
    #     counter = tf.Variable(0, dtype=tf.int32)
    #     adder = tf.assign(counter, counter + 1)
    #
    #     sliding_window_size = 3
    #
    #     ###### Placeholders
    #     learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
    #                                    name='learn_rate')
    #     keep_prob = tf.placeholder_with_default(1., shape=(),
    #                                             name='keep_prob')
    #
    #     # labels = tf.placeholder(dtype=tf.float32, shape=(None,),
    #     #                         name="labels")
    #
    #     ##### Placeholders for words
    #     word_pl = word_placeholders("word", self.max_grams, self.max_morph)
    #     neg_pl = [word_placeholders("neg_%d" % i, self.max_grams, self.max_morph) for i in range(self.n_neg)]
    #     context_pl = [word_placeholders("context_%d" % i, self.max_grams, self.max_morph) for i in
    #                   range(self.context_size)]
    #
    #     ##### Feature Embedding Matrices
    #     # gram_math = tf.get_variable("gram_emb_matr", shape=(self.gram_segmenter.unique_segments, self.h['gram_emb']))
    #     # morph_math = tf.get_variable("morph_emb_matr",
    #     #                              shape=(self.morph_segmenter.unique_segments, self.h['morph_emb']))
    #     # lemma_math = tf.get_variable("lemma_emb_matr",
    #     #                              shape=(self.lemma_segmenter.unique_segments, self.h['lemma_emb']))
    #
    #     emb_matr = tf.get_variable("emb_matr", shape=(self.segm_voc_size, self.h['feat_emb_size']))
    #
    #     ##### Embed Features
    #     word_feat = embed(emb_matr, word_pl)
    #     neg_feat_1 = [embed(emb_matr, neg) for neg in neg_pl]
    #     context_feat_1 = [embed(emb_matr, cont) for cont in context_pl]
    #
    #     ##### Embed Context
    #     with tf.variable_scope("context_embedding") as ce:
    #         context = tf.concat(
    #             [tf.expand_dims(cont_emb, axis=1)
    #              for cont_emb in context_feat_1],
    #             axis=1)
    #
    #         context_conv1 = convolutional_layer(context,
    #                                             self.h['d_out'],
    #                                             (sliding_window_size, context.shape[2]),
    #                                             keep_prob,
    #                                             tf.nn.sigmoid)
    #
    #         context_emb = tf.nn.l2_normalize(pooling_layer(context_conv1), axis=1)
    #
    #     ##### Word Embedding
    #
    #     with tf.variable_scope('word_projection') as wp:
    #         word_emb = tf.nn.l2_normalize(
    #             embedding_projection(word_feat, self.h['d_out'], self.h['d_out'], keep_prob)
    #         )
    #         wp.reuse_variables()
    #         neg_emb = [
    #             tf.nn.l2_normalize(
    #                 embedding_projection(neg_feat, self.h['d_out'], self.h['d_out'], keep_prob)
    #             )
    #             for neg_feat in neg_feat_1
    #         ]
    #
    #     ##### Loss
    #
    #     with tf.variable_scope("loss") as loss_context:
    #         context_word_sim = cosine_sim(context_emb, word_emb)
    #
    #         neg_context = tf.concat([
    #             tf.expand_dims(neg, axis=2) for neg in neg_emb
    #         ], axis=2)
    #
    #         neg_cont_sim = bulk_cosine_sim(context_emb, neg_context)
    #
    #         word_neg_sim_diff = tf.expand_dims(context_word_sim, axis=1) - neg_cont_sim
    #
    #         delta_sum = tf.reduce_sum(tf.math.exp(-self.temp * word_neg_sim_diff), axis=1)
    #         log_delta = tf.math.log(1. + delta_sum)
    #         loss = tf.reduce_sum(log_delta, name='loss_')
    #
    #     train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)
    #
    #     final = word_emb
    #
    #     saveloss = tf.summary.scalar('loss', loss)
    #
    #     self.terminals = {
    #         'context': context_pl,
    #         'words': [word_pl],
    #         'negative': neg_pl,
    #         # 'labels': labels,
    #         'loss': loss,
    #         'train': train,
    #         'adder': adder,
    #         'learning_rate': learning_rate,
    #         'batch_count': counter,
    #         'final': final,
    #         'saveloss': saveloss,
    #         'dropout': keep_prob
    #     }

    # def assemble_model_emb_based(self):
    #     counter = tf.Variable(0, dtype=tf.int32)
    #     adder = tf.assign(counter, counter + 1)
    #
    #     sliding_window_size = 3
    #
    #     ###### Placeholders
    #     learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
    #                                    name='learn_rate')
    #     keep_prob = tf.placeholder_with_default(1., shape=(),
    #                                             name='keep_prob')
    #
    #     ##### Placeholders for words
    #     num_words = self.context_size + 1 + self.n_neg
    #     feat_emb = self.h['feat_emb_size']
    #
    #     # words_pl = tf.placeholder(dtype=tf.int32, shape=(None, None))
    #     gram_pl = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_grams))
    #     morph_pl = tf.placeholder(dtype=tf.int32, shape=(None, None, self.max_morph))
    #     lemma_pl = tf.placeholder(dtype=tf.int32, shape=(None, None, self.lemma_segmenter.max_len))
    #
    #     # word_emb_matr = emb_matr_with_padding('word',
    #     #                                       shape=(self.vocabulary_size,
    #     #                                              self.h['feat_emb_size']))
    #     gram_emb_matr = emb_matr_with_padding('gram',
    #                                           shape=(self.gram_segmenter.unique_segments,
    #                                                  self.h['feat_emb_size']))
    #     morph_emb_matr = emb_matr_with_padding('morph',
    #                                            shape=(self.morph_segmenter.unique_segments,
    #                                                   self.h['feat_emb_size']))
    #     lemma_emb_matr = emb_matr_with_padding('lemma',
    #                                            shape=(
    #                                                self.lemma_segmenter.unique_segments + self.lemma_segmenter.total_words,
    #                                                self.h['feat_emb_size']))
    #
    #     # word_emb = tf.nn.embedding_lookup(word_emb_matr, words_pl, name='word_lookup')
    #     gram_emb = tf.reduce_sum(tf.nn.embedding_lookup(gram_emb_matr, gram_pl, name='gram_lookup'), axis=-2)
    #     morph_emb = tf.reduce_sum(tf.nn.embedding_lookup(morph_emb_matr, morph_pl, name='morph_lookup'), axis=-2)
    #     lemma_emb = tf.reduce_sum(tf.nn.embedding_lookup(lemma_emb_matr, lemma_pl, name='lemma_lookup'), axis=-2)
    #
    #     all_concat = tf.concat([
    #         # word_emb,
    #         gram_emb,
    #         morph_emb,
    #         lemma_emb
    #     ], axis=-1)
    #
    #     context = all_concat[:, :self.context_size, ...]
    #     word = all_concat[:, self.context_size, ...]
    #     neg = all_concat[:, self.context_size + 1:, ...]
    #
    #     # context = tf.concat([
    #     #     word_emb[:, :self.context_size, ...],
    #     #     gram_emb[:, :self.context_size, ...],
    #     #     morph_emb[:, :self.context_size, ...],
    #     #     lemma_emb[:, :self.context_size, ...]
    #     # ], axis=-1)
    #     # # shape (None, context_size, feat_emb * 3)
    #     #
    #     # word = tf.concat([
    #     #     word_emb[:, self.context_size, ...],
    #     #     gram_emb[:, self.context_size, ...],
    #     #     morph_emb[:, self.context_size, ...],
    #     #     lemma_emb[:, self.context_size, ...]
    #     # ], axis=-1)
    #     #
    #     # neg = tf.concat([
    #     #     word_emb[:, self.context_size + 1:, ...],
    #     #     gram_emb[:, self.context_size + 1:, ...],
    #     #     morph_emb[:, self.context_size + 1:, ...],
    #     #     lemma_emb[:, self.context_size + 1:, ...]
    #     # ], axis=-1)
    #
    #     ##### Embed Context
    #     with tf.variable_scope("context_embedding") as ce:
    #         context_conv1 = convolutional_layer(context,
    #                                             self.h['d_out'],
    #                                             (sliding_window_size, context.shape[2]),
    #                                             keep_prob,
    #                                             tf.nn.tanh)
    #
    #         context_emb = tf.nn.l2_normalize(pooling_layer(context_conv1), axis=1)
    #
    #     ##### Word Embedding
    #
    #     with tf.variable_scope('word_projection') as wp:
    #         word_emb = tf.nn.l2_normalize(tf.reshape(
    #             embedding_projection(tf.reshape(
    #                 word, (-1, feat_emb * 3)
    #             ), self.h['d_out'], self.h['d_out'], keep_prob), (-1, 1, self.h['d_out'])), axis=-1
    #         )
    #         wp.reuse_variables()
    #         neg_emb = tf.nn.l2_normalize(tf.reshape(
    #             embedding_projection(tf.reshape(
    #                 neg, (-1, feat_emb * 3)
    #             ), self.h['d_out'], self.h['d_out'], keep_prob), (-1, self.n_neg, self.h['d_out'])), axis=-1
    #         )
    #         all_emb = tf.nn.l2_normalize(tf.reshape(
    #             embedding_projection(tf.reshape(
    #                 all_concat, (-1, feat_emb * 3)
    #             ), self.h['d_out'], self.h['d_out'], keep_prob), (-1, self.h['d_out'])), axis=-1
    #         )
    #
    #     ##### Loss
    #
    #     with tf.variable_scope("loss") as loss_context:
    #         context_word_sim = cosine_sim(context_emb, word_emb)
    #
    #         neg_cont_sim = cosine_sim(context_emb, neg_emb)
    #
    #         word_neg_sim_diff = context_word_sim - neg_cont_sim
    #
    #         delta_sum = tf.reduce_sum(tf.math.exp(-self.temp * word_neg_sim_diff), axis=-1)
    #         log_delta = tf.math.log(1. + delta_sum)
    #         loss = tf.reduce_mean(log_delta, name='loss_')
    #
    #     train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)
    #
    #     ###### Final Emb
    #
    #     # with tf.variable_scope("final_embeddings") as fe:
    #     #     word_pl_final = tf.placeholder(dtype=tf.int32, shape=(None, ))
    #     #     gram_pl_final = tf.placeholder(dtype=tf.int32, shape=(None, self.max_grams))
    #     #     morph_pl_final = tf.placeholder(dtype=tf.int32, shape=(None, self.max_morph))
    #     #     lemma_pl_final = tf.placeholder(dtype=tf.int32, shape=(None, self.lemma_segmenter.max_len))
    #     #
    #     #     word_emb_final = tf.nn.embedding_lookup(word_emb_matr, word_pl_final, name='word_lookup_final')
    #     #     gram_emb_final = tf.reduce_sum(tf.nn.embedding_lookup(gram_emb_matr, gram_pl_final, name='gram_lookup_final'), axis=-2)
    #     #     morph_emb_final = tf.reduce_sum(tf.nn.embedding_lookup(morph_emb_matr, morph_pl_final, name='morph_lookup_final'), axis=-2)
    #     #     lemma_emb_final = tf.reduce_sum(tf.nn.embedding_lookup(lemma_emb_matr, lemma_pl_final, name='lemma_lookup_final'), axis=-2)
    #     #
    #     #     all = tf.concat([
    #     #         word_emb_final,
    #     #         gram_emb_final,
    #     #         morph_emb_final,
    #     #         lemma_emb_final
    #     #     ], axis=-1)
    #     #
    #     #     all_emb = tf.nn.l2_normalize(
    #     #         embedding_projection(
    #     #             all, self.h['d_out'], self.h['d_out'], keep_prob), axis=-1
    #     #     )
    #
    #     final = all_emb
    #
    #     saveloss = tf.summary.scalar('loss', loss)
    #
    #     self.terminals = {
    #         'placeholders': [gram_pl, morph_pl, lemma_pl],
    #         # 'final_placeholders': [word_pl_final, gram_pl_final, morph_pl_final, lemma_pl_final],
    #         'loss': loss,
    #         'train': train,
    #         'adder': adder,
    #         'learning_rate': learning_rate,
    #         'batch_count': counter,
    #         'final': final,
    #         'saveloss': saveloss,
    #         'dropout': keep_prob
    #     }

    def assemble_model(self):
        counter = tf.Variable(0, dtype=tf.int32)
        adder = tf.assign(counter, counter + 1)

        sliding_window_size = 3
        total_segments = self.max_morph + self.lemma_segmenter.max_len #+ self.max_grams

        ###### Placeholders
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                       name='learn_rate')
        keep_prob = tf.placeholder_with_default(1., shape=(),
                                                name='keep_prob')

        ##### Placeholders for words

        all_pl = tf.sparse.placeholder(dtype=tf.int32, shape=(None, None, total_segments), name="all_pl")
        final_pl = tf.sparse.placeholder(dtype=tf.int32, shape=(None, total_segments), name="final_pl")

        context = tf.sparse.slice(all_pl,
                                  [0, 0, 0],
                                  [self.batch_size, self.context_size, total_segments])
        word_neg = tf.sparse.slice(all_pl,
                                   [0, self.context_size, 0],
                                   [self.batch_size, self.n_neg + 1, total_segments])

        ##### CNN emb
        with tf.variable_scope("context_embedding") as ce:
            cnn_embedding = tf.get_variable("cnn_embedding", shape=(self.segm_voc_size, self.h['d_out']),
                                            dtype=tf.float32)
            cnn_bias = tf.get_variable("cnn_bias", shape=(self.h['d_out'],), dtype=tf.float32)

            contexts = []
            for i in range(0, self.context_size - sliding_window_size):
                c = tf.sparse.slice(context, [0, i, 0],
                                    [self.batch_size, sliding_window_size, total_segments])
                contexts.append(tf.nn.tanh(tf.expand_dims(tf.reduce_sum(
                    tf.nn.safe_embedding_lookup_sparse(
                        embedding_weights=cnn_embedding,
                        sparse_ids=c,
                        sparse_weights=None,
                        combiner="sum"), axis=-2) + tf.reshape(cnn_bias, (1, -1)), axis=1,
                                                          )))

            context_emb = tf.nn.l2_normalize(tf.reduce_max(tf.concat(
                contexts, axis=1
            ), axis=1))

        ##### Word Embedding

        with tf.variable_scope('word_projection') as wp:
            word_emb_h1 = tf.get_variable("word_emb_h1", shape=(self.segm_voc_size, self.h['d_hid']), dtype=tf.float32)
            bias_word_emb_h1 = tf.get_variable("bias_word_emb_h1", shape=(self.h['d_hid'],), dtype=tf.float32)

            flattenwith_batch = tf.sparse.reshape(word_neg, (-1, total_segments), name="batch_word_neg_flat")

            embed_layer_1 = tf.nn.tanh(
                tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=word_emb_h1,
                    sparse_ids=flattenwith_batch,
                    sparse_weights=None,
                    combiner="sum") + bias_word_emb_h1)

            embed_layer_2 = tf.layers.dense(
                embed_layer_1,
                self.h['d_out'],
                activation=tf.nn.tanh,
                name="word_emb_h2")

            embed_normalize = tf.nn.l2_normalize(
                embed_layer_2,
                axis=-1)

            batch_word_neg_restored = tf.reshape(embed_normalize,
                                                 shape=(-1, self.n_neg + 1, self.h['d_out']),
                                                 name="batch_word_neg_restored")

            word_emb = batch_word_neg_restored[:, 0, :]
            neg_emb = batch_word_neg_restored[:, 1:, :]

            wp.reuse_variables()

            final_embed_layer_1 = tf.nn.tanh(
                tf.nn.safe_embedding_lookup_sparse(
                    embedding_weights=word_emb_h1,
                    sparse_ids=final_pl,
                    sparse_weights=None,
                    combiner="mean") + bias_word_emb_h1)

            final_embed_layer_2 = tf.layers.dense(
                final_embed_layer_1,
                self.h['d_out'],
                activation=tf.nn.tanh,
                name="word_emb_h2")

        ##### Loss

        with tf.variable_scope("loss") as loss_context:
            context_word_sim = cosine_sim(context_emb, word_emb)

            print()

            neg_cont_sim = cosine_sim(tf.expand_dims(context_emb, axis=1), neg_emb)

            word_neg_sim_diff = tf.expand_dims(context_word_sim, axis=1) - neg_cont_sim

            delta_sum = tf.reduce_sum(tf.math.exp(-self.temp * word_neg_sim_diff), axis=-1)
            log_delta = tf.math.log(1. + delta_sum)
            loss = tf.reduce_mean(log_delta, name='loss_')

        train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        final = tf.nn.l2_normalize(final_embed_layer_2, axis=1)

        saveloss = tf.summary.scalar('loss', loss)

        self.terminals = {
            'placeholders': all_pl,
            'final_pl': final_pl,
            'loss': loss,
            'train': train,
            'adder': adder,
            'learning_rate': learning_rate,
            'batch_count': counter,
            'final': final,
            'saveloss': saveloss,
            'dropout': keep_prob
        }

    def update(self, batch, lr=0.001):
        train_ = self.terminals['train']
        adder_ = self.terminals['adder']
        feed_dict = self.prepare_batch(batch, learn_rate=lr, keep_prob=0.95)
        _, batch_count = self.sess.run([train_, adder_], feed_dict=feed_dict)

    # def prepare_batch_orig(self, batch, learn_rate, keep_prob):
    #     context = batch[:, :self.context_size]
    #     word = batch[:, self.context_size, None]
    #     neg = batch[:, self.context_size + 1:]
    #
    #     cont_pl = self.terminals['context']
    #     words_pl = self.terminals['words']
    #     neg_pl = self.terminals['negative']
    #
    #     feed_dict = {
    #         self.terminals['learning_rate']: learn_rate,
    #         self.terminals['dropout']: keep_prob
    #     }
    #
    #     for placeholder, data in zip([cont_pl, words_pl, neg_pl], [context, word, neg]):
    #         for ind, pl in enumerate(placeholder):
    #             sparse_segments = self.prepare_for_placeholder(data[:, ind])
    #             feed_dict.update(dict(zip(pl, sparse_segments)))
    #     return feed_dict

    def prepare_batch(self, batch, learn_rate=0.0, keep_prob=1., saving=False):

        placeholders = self.terminals['final_pl'] if saving else self.terminals['placeholders']
        n_words = batch.shape[1]
        n_batch = batch.shape[0]

        feed_dict = {
            self.terminals['learning_rate']: learn_rate,
            self.terminals['dropout']: keep_prob
        }

        segmenters = [self.lemma_segmenter, self.gram_segmenter, self.morph_segmenter]
        # segmenters = [self.lemma_segmenter, self.morph_segmenter]
        # offset = [
        #     0,
        #     segmenters[0].unique_segments,
        #     segmenters[0].unique_segments + segmenters[1].unique_segments
        # ]
        pos_offset = [
            0,
            segmenters[0].max_len,
            segmenters[0].max_len + segmenters[1].max_len,
            segmenters[0].max_len + segmenters[1].max_len + segmenters[2].max_len
        ]

        indices_ = []
        values_ = []

        lim = self.max_morph + self.lemma_segmenter.max_len + self.max_grams

        bbatch = np.zeros((n_batch, n_words, lim))

        for i in range(n_words):
            bbatch[:, i, pos_offset[0]:pos_offset[1]] = segmenters[0].segment(batch[:, i])
            bbatch[:, i, pos_offset[1]:pos_offset[2]] = segmenters[0].segment(batch[:, i])
            bbatch[:, i, pos_offset[2]:pos_offset[3]] = segmenters[0].segment(batch[:, i])

        restricted = [segmenter.padding for segmenter in segmenters]

        indices = np.where((bbatch != restricted[0]) & (bbatch != restricted[1]) & (bbatch != restricted[2]))
        # indices = np.where((bbatch != restricted[0]) & (bbatch != restricted[1]))
        values = bbatch[indices]
        shape = bbatch.shape

        ind = np.vstack(indices).T

        # for b in range(n_batch):
        #     for i in range(n_words):
        #         c_pos = 0
        #         for ind, segmenter in enumerate(segmenters):
        #             seg_ids = np.array(segmenter.w2s[batch[b, i]], dtype=np.int32).reshape((-1, 1)) + \
        #                       offset[ind]
        #             app = seg_ids[:lim - c_pos]
        #             b_id = np.full((app.size, 1), b)
        #             i_id = np.full((app.size, 1), i)
        #             s_id = np.arange(c_pos, c_pos + app.size, 1).reshape(-1, 1)
        #             c_pos += app.size
        #             indices_.append(np.hstack([b_id, i_id, s_id]))
        #             values_.append(app)

        # indices = np.vstack(indices_)
        # values = np.concatenate(values_)
        # shape = (batch.shape[0], batch.shape[1], lim)

        if saving:
            ind = np.delete(ind, 1, axis=1)
            shape = (batch.shape[0], lim)

        feed_dict[placeholders] = tf.SparseTensorValue(ind,
                                                       values.reshape((-1,)),
                                                       shape)

        return feed_dict

    def evaluate(self, batch, save=False):
        feed_dict = self.prepare_batch(batch, 0.)

        loss_val, summary, batch_count = self.sess.run([self.terminals['loss'],
                                                        self.terminals['saveloss'],
                                                        self.terminals['batch_count']],
                                                       feed_dict=feed_dict)

        self.summary_writer.add_summary(summary, batch_count)

        if save:
            self.saver.save(self.sess, self.ckpt_path)

        return loss_val, batch_count

    def embed_words(self):
        final = []
        batch_size = 50
        for offset in range(0, self.vocabulary_size, batch_size):
            feed_dict = self.prepare_batch(
                np.array(list(range(offset, min(offset + batch_size, self.vocabulary_size))), dtype=np.int32).reshape(
                    (-1, 1)), saving=True)
            embs = self.sess.run(self.terminals['final'], feed_dict)
            final.append(embs)

        return np.vstack(final)

    # def prepare_for_placeholder(self, entry):
    #     return [
    #         denseNDArrayToSparseTensor(segmenter.segment(entry),
    #                                    segmenter.padding)
    #         for segmenter in [self.gram_segmenter,
    #                           self.morph_segmenter,
    #                           self.lemma_segmenter]
    #     ]


def emb_matr_with_padding(prefix, shape):
    h = shape[0] - 1
    w = shape[1]
    m = tf.get_variable(prefix + "_emb_matr", shape=(h, w), dtype=tf.float32)
    p = tf.zeros(shape=(1, w), dtype=tf.float32)
    return tf.concat([m, p], axis=0, name=prefix + "_emb_matr_pad")


def convolutional_layer(input, units, cnn_kernel_shape, kp, activation=None):
    # padded = tf.pad(input, tf.constant([[0, 0], [1, 1], [0, 0]]))
    emb_sent_exp = tf.expand_dims(input, axis=3)
    # emb_sent_exp_drop = tf.nn.dropout(emb_sent_exp, keep_prob=kp)
    convolve = tf.layers.conv2d(emb_sent_exp,
                                units,
                                cnn_kernel_shape,
                                use_bias=False,
                                activation=activation,
                                data_format='channels_last')
    return convolve


def pooling_layer(input):
    return tf.reduce_max(input, axis=1)
    # pool = tf.nn.max_pool(input, ksize=[1, input.shape[1], 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    # return tf.reshape(pool, (-1, 1, input.shape[3]))


Subword = namedtuple('Subword', 'gram morph lemma')


def word_placeholders(postfix, max_gram_segments, max_morph_segments):
    gram_segments = tf.sparse.placeholder(dtype=tf.int32, shape=(None, max_gram_segments),
                                          name="in_grams_%s" % postfix)
    morph_segments = tf.sparse.placeholder(dtype=tf.int32, shape=(None, max_morph_segments),
                                           name="in_morph_%s" % postfix)
    lemma_segments = tf.sparse.placeholder(dtype=tf.int32, shape=(None, 1),
                                           name="in_lemma_%s" % postfix)

    return Subword(gram_segments, morph_segments, lemma_segments)


def embed(emb_matr, subword):
    def embed_feat(feat):
        return tf.nn.embedding_lookup_sparse(emb_matr,
                                             feat,
                                             sp_weights=None,
                                             combiner='sum')

    gram_emb = embed_feat(subword.gram)
    morph_emb = embed_feat(subword.morph)
    lemma_emb = embed_feat(subword.lemma)

    return tf.concat([gram_emb, morph_emb, lemma_emb], axis=1)


# def embedding_projection(input_, emb_h1, emb_h2, kp):
#     # input_drop = tf.nn.dropout(input_, keep_prob=kp, name="drop_1")
#     h1 = tf.layers.dense(input_, emb_h1, activation=tf.nn.sigmoid, name='projection_layer_1')
#     # h1_drop = tf.nn.dropout(h1, keep_prob=kp, name="drop_2")
#     h2 = tf.layers.dense(h1, emb_h2, activation=tf.nn.sigmoid, name='projection_layer_2')
#     return h2


def cosine_sim(emb1, emb2):
    return tf.reduce_sum(emb1 * emb2, axis=-1)

# def bulk_cosine_sim(single, context):
#     return tf.reduce_sum(tf.expand_dims(single, axis=2) * context, axis=1)

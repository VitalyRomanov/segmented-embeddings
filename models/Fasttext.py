from models import Skipgram
from utils import WordSegmenter
import tensorflow as tf
import numpy as np
import pickle

class Fasttext(Skipgram):
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 segmenter_path=None,
                 max_segments=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None,
                 model_name='fasttext',
                 model_type='sum'):
        self.graph_path = graph_path
        self.vocabulary_size = vocab_size
        self.emb_size = emb_size
        self.model_name = model_name + "_" + model_type
        self.ckpt_path = ckpt_path
        self.model_type = model_type

        self.segmenter = WordSegmenter(segmenter_path, max_segments, vocab_size)

        self.max_segments = self.segmenter.max_len

        self.assemble_model()

        self.open_session(gpu_options)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.graph_path,
                                                    graph=self.sess.graph)

    def assemble_model(self):
        if self.model_type == "sum":
            self.sum_model()
        elif self.model_type == "flat":
            self.flat_model()
        else:
            raise NotImplementedError()

    def sum_model(self):
        counter = tf.Variable(0, dtype=tf.int32)
        adder = tf.assign(counter, counter + 1)

        ###### Placeholders
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                       name='learn_rate')

        labels = tf.placeholder(dtype=tf.float32, shape=(None,),
                                name="labels")

        in_segments = tf.placeholder(dtype=tf.int32, shape=(None, self.max_segments),
                                     name="in_words")

        out_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                   name="out_words")

        ###### Input word embedding matrix
        segm_matr = tf.get_variable("SEGM",
                                    shape=[self.vocabulary_size + \
                                           self.segmenter.unique_segments - 1,
                                           # minus 1 to add constant zero padding for short words
                                           self.emb_size],
                                    dtype=tf.float32)

        padding = tf.zeros(shape=(1, self.emb_size), dtype=tf.float32)

        embedding_matr = tf.concat([segm_matr, padding], axis=0)

        ###### Output word embedding matrix
        out_matr = tf.get_variable("OUT", shape=[self.vocabulary_size, self.emb_size],
                                   dtype=tf.float32)

        out_bias = tf.get_variable("out_bias", dtype=tf.float32,
                                   initializer=tf.zeros([self.vocabulary_size]))

        ###### Embeddings
        segm_emb = tf.nn.embedding_lookup(embedding_matr, in_segments, name="in_lookup")
        in_emb = tf.reduce_sum(segm_emb, axis=1, name="sum_segm_emb")

        out_emb = tf.nn.embedding_lookup(out_matr, out_words,
                                         name="out_lookup")

        ###### Calculate loss
        bias_slice = tf.gather_nd(out_bias, tf.reshape(out_words, (-1, 1)))

        logits = tf.reduce_sum(in_emb * out_emb,
                               axis=1, name="inner_product") + bias_slice

        per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        loss = tf.reduce_mean(per_item_loss, axis=0)

        ###### Extra

        train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        final = tf.nn.l2_normalize(in_emb, axis=1)

        saveloss = tf.summary.scalar('loss', loss)

        self.terminals = {
            'in_words': in_segments,
            'out_words': out_words,
            'labels': labels,
            'loss': loss,
            'train': train,
            'adder': adder,
            'learning_rate': learning_rate,
            'batch_count': counter,
            'final': final,
            'saveloss': saveloss
        }

    def flat_model(self):
        counter = tf.Variable(0, dtype=tf.int32)
        adder = tf.assign(counter, counter + 1)

        ###### Placeholders
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                       name='learn_rate')

        labels = tf.placeholder(dtype=tf.float32, shape=(None,),
                                name="labels")

        in_segments = tf.placeholder(dtype=tf.int32, shape=(None,),
                                  name="in_words")

        out_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                   name="out_words")

        ###### Input word embedding matrix
        segm_matr = tf.get_variable("SEGM",
                                    shape=[self.vocabulary_size + \
                                           self.segmenter.unique_segments - 1,
                                           # minus 1 to add constant zero padding for short words
                                           self.emb_size],
                                  dtype=tf.float32)

        padding = tf.zeros(shape=(1, self.emb_size), dtype=tf.float32)

        embedding_matr = tf.concat([segm_matr, padding], axis=0)

        ###### Output word embedding matrix
        out_matr = tf.get_variable("OUT", shape=[self.vocabulary_size, self.emb_size],
                                   dtype=tf.float32)

        out_bias = tf.get_variable("out_bias", dtype=tf.float32,
                                   initializer=tf.zeros([self.vocabulary_size]))

        ###### Embeddings
        in_emb = tf.nn.embedding_lookup(embedding_matr, in_segments, name="in_lookup")

        out_emb = tf.nn.embedding_lookup(out_matr, out_words,
                                         name="out_lookup")

        ###### Calculate loss
        bias_slice = tf.gather_nd(out_bias, tf.reshape(out_words, (-1, 1)))

        logits = tf.reduce_sum(in_emb * out_emb,
                               axis=1, name="inner_product") + bias_slice

        per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        loss = tf.reduce_mean(per_item_loss, axis=0)

        ###### Extra

        train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        final = tf.nn.l2_normalize(in_emb, axis=1)

        in_word_segm = tf.placeholder(dtype=tf.int32, shape=(None, self.max_segments),
                                name="word_segments")

        s_e = tf.nn.embedding_lookup(embedding_matr, in_word_segm)

        sum_emb = tf.nn.l2_normalize(tf.reduce_sum(s_e, axis=1))
        mean_emb = tf.nn.l2_normalize(tf.reduce_mean(s_e, axis=1))

        saveloss = tf.summary.scalar('loss', loss)

        self.terminals = {
            'in_words': in_segments,
            'out_words': out_words,
            'labels': labels,
            'loss': loss,
            'train': train,
            'adder': adder,
            'learning_rate': learning_rate,
            'batch_count': counter,
            'final': final,
            'saveloss': saveloss,
            'in_word_segm': in_word_segm,
            'sum_emb': sum_emb,
            'mean_emb': mean_emb
        }

    def expand_ids(self, ids):
        return self.segmenter.segment(ids)

    def prepare_batch(self, batch):

        if self.model_type == "sum":
            return self.expand_ids(batch[:, 0]), batch[:, 1], batch[:, 2]

        elif self.model_type == "flat":
            batch_size = batch.shape[0]
            in_, out_, lbl_ = self.expand_ids(batch[:, 0]), batch[:, 1], batch[:, 2]
            lens = self.segmenter.get_lens(batch[:, 0])

            in_b = []
            out_b = []
            lbl_b = []

            for i in range(batch_size):
                segments = in_[i, :lens[i]]
                out_words = np.full_like(segments, out_[i])
                labels = np.full_like(segments, lbl_[i])

                in_b.append(segments)
                out_b.append(out_words)
                lbl_b.append(labels)

            return np.concatenate(in_b, axis=0), np.concatenate(out_b, axis=0), np.concatenate(lbl_b, axis=0)

    def save_snapshot(self):
        if self.model_type == sum:
            batch_count = self.sess.run(self.terminals['batch_count'])
            path = "./%s/%s_%d_%d" % (self.graph_path,
                                      self.model_name,
                                      self.vocabulary_size,
                                      batch_count)
            ckpt_p = "%s/model.ckpt" % path

            print("\nDumpung vocabulary of size %d\n" % self.vocabulary_size)

            final = self.embed_words()

            emb_dump_path = "./embeddings/%s_%d.pkl" % (self.model_name, self.vocabulary_size)
            pickle.dump(final, open(emb_dump_path, "wb"))

            _ = self.saver.save(self.sess, ckpt_p)

        elif self.model_type == 'flat':
            batch_count = self.sess.run(self.terminals['batch_count'])
            path = "./%s/%s_%d_%d" % (self.graph_path,
                                      self.model_name,
                                      self.vocabulary_size,
                                      batch_count)
            ckpt_p = "%s/model.ckpt" % path

            print("\nDumpung vocabulary of size %d\n" % self.vocabulary_size)

            final_sum = []
            final_mean = []
            batch_size = 50
            for offset in range(0, self.vocabulary_size, batch_size):
                ids = self.expand_ids(
                    np.array(list(range(offset, min(offset + batch_size, self.vocabulary_size))), dtype=np.int32))
                sum_embs, mean_emb = self.sess.run([self.terminals['sum_emb'], self.terminals['mean_emb']],
                                                   {self.terminals['in_word_segm']: ids})
                final_sum.append(sum_embs)
                final_mean.append(mean_emb)

            final_sum = np.vstack(final_sum)
            final_mean = np.vstack(final_mean)

            emb_dump_path = "./embeddings/%s_%d_sum.pkl" % (self.model_name, self.vocabulary_size)
            pickle.dump(final_sum, open(emb_dump_path, "wb"))
            emb_dump_path = "./embeddings/%s_%d_mean.pkl" % (self.model_name, self.vocabulary_size)
            pickle.dump(final_mean, open(emb_dump_path, "wb"))

            _ = self.saver.save(self.sess, ckpt_p)
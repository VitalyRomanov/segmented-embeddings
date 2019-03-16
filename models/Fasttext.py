from models import Skipgram
from utils import WordSegmenter
import tensorflow as tf
import numpy as np

class Fasttext(Skipgram):
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 segmenter_path=None,
                 max_segments=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None,
                 model_name='fasttext'):
        self.graph_path = graph_path
        self.vocabulary_size = vocab_size
        self.emb_size = emb_size
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.segmenter = WordSegmenter(segmenter_path, max_segments, vocab_size)

        self.max_segments = self.segmenter.max_len
        self.segm_voc_size = self.vocabulary_size + self.segmenter.unique_segments - 1
        # minus 1 to add constant zero padding for short words

        self.assemble_model()

        self.open_session(gpu_options)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.graph_path,
                                                    graph=self.sess.graph)

    def assemble_model(self):
        counter = tf.Variable(0, dtype=tf.int32)
        adder = tf.assign(counter, counter + 1)

        ###### Placeholders
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                       name='learn_rate')

        labels = tf.placeholder(dtype=tf.float32, shape=(None,),
                                name="labels")

        # in_segments = tf.placeholder(dtype=tf.int32, shape=(None, self.max_segments),
        #                              name="in_words")
        in_segments = tf.sparse.placeholder(dtype=tf.int32, shape=(None, self.max_segments),
                                            name="in_words")

        out_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                   name="out_words")

        ###### Input word embedding matrix
        segm_matr = tf.get_variable("SEGM",
                                    shape=[self.segm_voc_size, self.emb_size],
                                    dtype=tf.float32)

        padding = tf.zeros(shape=(1, self.emb_size), dtype=tf.float32)

        embedding_matr = tf.concat([segm_matr, padding], axis=0)

        ###### Output word embedding matrix
        out_matr = tf.get_variable("OUT", shape=[self.vocabulary_size, self.emb_size],
                                   dtype=tf.float32)

        out_bias = tf.get_variable("out_bias", dtype=tf.float32,
                                   initializer=tf.zeros([self.vocabulary_size]))

        ###### Embeddings
        in_emb = tf.nn.embedding_lookup_sparse(embedding_matr,
                                                 in_segments,
                                                 sp_weights=None,
                                                 combiner='sum',
                                                 name="sum_segm_emb")
        # segm_emb = tf.nn.embedding_lookup(embedding_matr, in_segments, name="in_lookup")
        # in_emb = tf.reduce_sum(segm_emb, axis=1, name="sum_segm_emb")

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

    def expand_ids(self, ids):
        # return self.segmenter.segment(ids)
        return denseNDArrayToSparseTensor(self.segmenter.segment(ids), self.segmenter.padding)

def denseNDArrayToSparseTensor(arr, ignore_val):
  idx  = np.where(arr != ignore_val)
  return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)

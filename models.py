import tensorflow as tf
import pickle
import numpy as np
import sys
from tensorflow import GPUOptions


class Skipgram:
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None):

        self.graph_path = graph_path
        self.vocabulary_size = vocab_size
        self.emb_size = emb_size
        self.model_name = 'skipgram'
        self.ckpt_path = ckpt_path

        self.assemble_model()

        self.open_session(gpu_options)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.graph_path,
                                               graph=self.sess.graph)

    def __del__(self):
        self.sess.close()

    def assemble_model(self):
        counter = tf.Variable(0, dtype=tf.int32)
        adder = tf.assign(counter, counter + 1)

        learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                       name='learn_rate')

        labels = tf.placeholder(dtype=tf.float32, shape=(None,),
                                name="labels")

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                  name="in_words")

        out_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                   name="out_words")

        in_matr = tf.get_variable("IN", shape=[self.vocabulary_size, self.emb_size],
                                  dtype=tf.float32)

        out_matr = tf.get_variable("OUT", shape=[self.vocabulary_size, self.emb_size],
                                   dtype=tf.float32)

        out_bias = tf.get_variable("out_bias", dtype=tf.float32,
                                   initializer=tf.zeros([self.vocabulary_size]))

        in_emb = tf.nn.embedding_lookup(in_matr, in_words, name="in_lookup")

        out_emb = tf.nn.embedding_lookup(out_matr, out_words,
                                         name="out_lookup")
        bias_slice = tf.gather_nd(out_bias, tf.reshape(out_words, (-1, 1)))

        logits = tf.reduce_sum(in_emb * out_emb,
                               axis=1, name="inner_product") + bias_slice

        per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        loss = tf.reduce_mean(per_item_loss, axis=0)

        train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        final = tf.nn.l2_normalize(in_emb, axis=1)

        saveloss = tf.summary.scalar('loss', loss)

        self.terminals = {
            'in_words': in_words,
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

    def open_session(self, gpu_options):
        if gpu_options:
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def restore_graph(self, path=None):
        try:
            if not path:
                path = self.ckpt_path
            self.saver.restore(self.sess, path)
            self.sess.graph.as_default()
        except:
            print("Cannot restore: checkpoint does not exist")
            sys.exit()

    def save_snapshot(self):
        batch_count = self.sess.run(self.terminals['batch_count'])
        path = "./%s/%s_%d_%d" % (self.graph_path,
                                  self.model_name,
                                  self.vocabulary_size,
                                  batch_count)
        ckpt_p = "%s/model.ckpt" % path

        print("\nDumpung vocabulary of size %d\n" % self.vocabulary_size)
        ids = np.array(list(range(self.vocabulary_size)))

        emb_dump_path = "./embeddings/%s_%d.pkl" % (self.model_name, self.vocabulary_size)
        final = self.sess.run(self.terminals['final'], {self.terminals['in_words']: ids})
        pickle.dump(final, open(emb_dump_path, "wb"))

        _ = self.saver.save(self.sess, ckpt_p)

    def update(self, batch, lr=0.001):
        train_ = self.terminals['train']
        adder_ = self.terminals['adder']
        _, batch_count = self.sess.run([train_, adder_], feed_dict={
            self.terminals['in_words']: batch[:, 0],
            self.terminals['out_words']: batch[:, 1],
            self.terminals['labels']: batch[:, 2],
            self.terminals['learning_rate']: lr,
        })

    def evaluate(self, batch, save=False):
        loss_val, summary, batch_count = self.sess.run([self.terminals['loss'], self.terminals['saveloss'], self.terminals['batch_count']], feed_dict={
            self.terminals['in_words']: batch[:, 0],
            self.terminals['out_words']: batch[:, 1],
            self.terminals['labels']: batch[:, 2],
        })

        self.summary_writer.add_summary(summary, batch_count)

        if save:
            self.saver.save(self.sess, self.ckpt_path)

        return loss_val, batch_count



def assemble_graph(model='skipgram',
                   vocab_size=None,
                   emb_size=None,
                   segment_vocab_size=None,
                   max_word_segments=None):

    counter = tf.Variable(0, dtype=tf.int32)
    adder = tf.assign(counter, counter + 1)

    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learn_rate')
    labels = tf.placeholder(dtype=tf.float32, shape=(None,), name="labels")
    dropout = tf.placeholder(1.0, shape=())

    # we always have word embedding matrix
    in_matr = tf.get_variable("IN", shape=[vocab_size, emb_size],
                              dtype=tf.float32) #,
                              #initializer=tf.random_uniform([vocab_size, emb_size], -1.0, 1.0))

    ## Out matrix is the same across models
    out_matr = tf.get_variable("OUT", shape=[vocab_size, emb_size],
                               dtype=tf.float32) #,
                               #initializer=tf.truncated_normal([vocab_size, emb_size], stddev=1.0 / np.sqrt(emb_size)))
    out_bias = tf.get_variable("out_bias", dtype=tf.float32, initializer=tf.zeros([vocab_size]))
    out_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="out_words")
    out_emb = tf.nn.embedding_lookup(out_matr, out_words, name="out_lookup")
    bias_slice = tf.gather_nd(out_bias, tf.reshape(out_words, (-1, 1)))

    emb_segments_in_attention_mask = None

    if model == 'skipgram':

        assert vocab_size is not None
        assert emb_size is not None

        # embedding matrices

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="in_words")

        in_emb = tf.nn.embedding_lookup(in_matr, in_words, name="in_lookup")

    elif model == 'fasttext' or model == 'morph':

        assert vocab_size is not None
        assert segment_vocab_size is not None
        assert max_word_segments is not None
        assert emb_size is not None

        pad = tf.zeros(shape=(1,emb_size), name="padding_vector", dtype=tf.float32)

        segment_in_matr = tf.get_variable("SEGM_IN",
                                          dtype=tf.float32,
                                          initializer=tf.random_uniform([segment_vocab_size, emb_size], -1.0, 1.0))

        in_embedding_matrix = tf.concat([in_matr, segment_in_matr, pad], axis=0)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,max_word_segments), name="in_words")

        in_emb = tf.reduce_sum(tf.nn.embedding_lookup(in_embedding_matrix, in_words), axis=1)

    elif model == "attentive":

        assert vocab_size is not None
        assert segment_vocab_size is not None
        assert max_word_segments is not None
        assert emb_size is not None

        attentive_seq_len = max_word_segments

        pad = tf.zeros(shape=(1, emb_size), name="padding_vector", dtype=tf.float32)

        segment_in_matr = tf.get_variable("SEGM_IN", shape=(segment_vocab_size - 1, emb_size), dtype=tf.float32)

        in_embedding_matrix = tf.concat([in_matr, segment_in_matr, pad], axis=0)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None, attentive_seq_len), name="in_words")

        emb_segments_in = tf.nn.embedding_lookup(in_embedding_matrix, in_words)

        emb_segments_in_r = tf.reshape(emb_segments_in, (-1, attentive_seq_len * emb_size))

        def attention_layer(input_):
            d_out = tf.nn.dropout(input_, keep_prob=dropout)
            joined_attention = tf.layers.dense(d_out,
                                               attentive_seq_len,
                                               name='joined_attention',
                                               activation=tf.nn.sigmoid,
                                               kernel_regularizer=tf.nn.l2_loss)

            attention_mask = tf.reshape(joined_attention, (-1, attentive_seq_len, 1), name='attention_mask')
            soft_attention = tf.nn.softmax(attention_mask, axis=1, name='soft_attention_mask')
            return soft_attention

        with tf.variable_scope('attention') as att_scope:
            emb_segments_in_attention_mask = attention_layer(emb_segments_in_r)

        in_emb = tf.reduce_sum(emb_segments_in * emb_segments_in_attention_mask, axis=1)

    else:
        raise NotImplementedError("Invalid model name: %s" % model)

    final = tf.nn.l2_normalize(in_emb, axis=1)

    #logits = tf.layers.dense(in_emb, 50001)
    #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(out_words, 50001)))
    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product") + bias_slice
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_mean(per_item_loss, axis=0)
    #loss = tf.reduce_mean(
    #    tf.nn.nce_loss(
    #        weights=out_matr,
    #        biases=out_bias,
    #        labels=tf.reshape(out_words,(-1,1)),
    #        inputs=in_emb,
    #        num_sampled=20,
    #        num_classes=50001))

    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)
    #train = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    #train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    #opt = tf.train.GradientDescentOptimizer(learning_rate)
    #opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate)
    #tvars = tf.trainable_variables()
    #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), .1)
    #train = opt.apply_gradients(zip(grads, tvars))

    return {
        'in_words': in_words,
        'out_words': out_words,
        'labels': labels,
        'loss': loss,
        'train': train,
        'adder': adder,
        'learning_rate': learning_rate,
        'batch_count': counter,
        'final': final,
        'dropout': dropout,
        'attention_mask': emb_segments_in_attention_mask
    }


def assign_embeddings(sess, terminals, args, segmenter=None):
    in_words_ = terminals['in_words']
    final_ = terminals['final']
    dropout_ = terminals['dropout']
    attention_ = terminals['attention_mask']
    
    m_name = args['model_name']
    v_s = args['vocabulary_size']

    print("\nDumpung vocabulary of size %d\n" % v_s)
    ids = np.array(list(range(v_s)))

    if m_name in ['morph', 'fasttext']:
        ids_expanded = segmenter.segment(ids)

        emb_sum_path = "./embeddings/%s_%d_sum.pkl" % (m_name, v_s)
        final_sum = sess.run(final_, {in_words_: ids_expanded, dropout_: 1.0})
        pickle.dump(final_sum, open(emb_sum_path, "wb"))

        emb_voc_path = "./embeddings/%s_%d_voc.pkl" % (m_name, v_s)
        id_voc = np.zeros_like(ids_expanded)
        id_voc[:,0] = ids
        final_voc = sess.run(final_, {in_words_: id_voc, dropout_: 1.0})
        pickle.dump(final_voc, open(emb_voc_path, "wb"))

    if m_name == 'skipgram':
        emb_dump_path = "./embeddings/%s_%d.pkl" % (m_name, v_s)
        final = sess.run(final_, {in_words_: ids,
                              dropout_: 1.0})
        pickle.dump(final, open(emb_dump_path, "wb"))

    if m_name == 'attentive':
        sgm_p = args['segmenter'].split("/")[0]
        emb_dump_path = "./embeddings/%s_%s_%d.pkl" % (m_name, sgm_p, v_s)
        dump_path = "./embeddings/attention_mask_%s_%s_%d.pkl" % (sgm_p, m_name, v_s)

        attention_mask = sess.run(attention_, {in_words_: ids_expanded,
                              dropout_: 1.0})
        pickle.dump(attention_mask, open(dump_path, "wb"))
        pickle.dump(final, open(emb_dump_path, "wb"))
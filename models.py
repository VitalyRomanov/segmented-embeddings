import tensorflow as tf
import pickle
import numpy as np

def assemble_graph(model='skipgram',
                   vocab_size=None,
                   emb_size=None,
                   segment_vocab_size=None):

    counter = tf.Variable(0, dtype=tf.int32)
    adder = tf.assign(counter, counter + 1)

    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learn_rate')
    labels = tf.placeholder(dtype=tf.float32, shape=(None,), name="labels")

    ## OUT matrix is the same across models
    out_matr = tf.get_variable("OUT", shape=(vocab_size, emb_size), dtype=tf.float32)
    out_bias = tf.get_variable("out_bias", shape=(vocab_size,), dtype=tf.float32)
    out_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="out_words")
    out_emb = tf.nn.embedding_lookup(out_matr, out_words)
    bias_slice = tf.gather_nd(out_bias, tf.reshape(out_words, (-1,1)))

    ## IN placeholder is always the same
    in_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="in_words")

    if model == 'skipgram':

        assert vocab_size is not None
        assert emb_size is not None

        in_matr = tf.get_variable("IN", shape=(vocab_size, emb_size), dtype=tf.float32)

    elif model == 'fasttext' or model == 'morph':

        assert vocab_size is not None
        assert segment_vocab_size is not None
        assert emb_size is not None

        in_matr = tf.get_variable("IN", shape=(vocab_size + segment_vocab_size, emb_size), dtype=tf.float32)

    else:
        raise NotImplementedError("Invalid model name: %s" % model)

    in_emb = tf.nn.embedding_lookup(in_matr, in_words)

    final = tf.nn.l2_normalize(in_emb, axis=1)

    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product") + bias_slice
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_sum(per_item_loss, axis=0)

    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return {
        'in_words': in_words,
        'out_words': out_words,
        'labels': labels,
        'loss': loss,
        'train': train,
        'adder': adder,
        'learning_rate': learning_rate,
        'batch_count': counter,
        'final': final
    }

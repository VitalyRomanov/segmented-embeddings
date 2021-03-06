import tensorflow as tf
import pickle
import numpy as np

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
    in_matr = tf.get_variable("IN",
                              dtype=tf.float32,
                              initializer=tf.random_uniform([vocab_size, emb_size], -1.0, 1.0))

    ## Out matrix is the same across models
    out_matr = tf.get_variable("OUT",
                               dtype=tf.float32,
                               initializer=tf.truncated_normal([vocab_size, emb_size], stddev=1.0 / np.sqrt(emb_size)))
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

    final = tf.nn.l2_normalize(in_emb, axis=1)

    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product") + bias_slice
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_sum(per_item_loss, axis=0)

    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    ### nce loss
    # loss = tf.reduce_mean(
    #         tf.nn.nce_loss(
    #             weights=out_matr,
    #             biases=out_bias,
    #             labels=tf.reshape(out_words,(-1,1)),
    #             inputs=in_emb,
    #             num_sampled=20,
    #             num_classes=50001))

    ### optimization with clipping gradients
    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 30.)
    # train = opt.apply_gradients(zip(grads, tvars))

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
        'dropout': dropout
    }

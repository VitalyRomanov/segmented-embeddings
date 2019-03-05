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
    in_matr = tf.get_variable("IN", shape=(vocab_size, emb_size), dtype=tf.float32)

    ## Out matrix is the same across models
    out_matr = tf.get_variable("OUT", shape=(vocab_size, emb_size), dtype=tf.float32)
    out_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="out_words")
    out_emb = tf.nn.embedding_lookup(out_matr, out_words)

    emb_segments_in_attention_mask = None

    if model == 'skipgram':

        assert vocab_size is not None
        assert emb_size is not None

        # embedding matrices

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="in_words")

        in_emb = tf.nn.embedding_lookup(in_matr, in_words)

    elif model == 'fasttext' or model == 'morph':

        assert vocab_size is not None
        assert segment_vocab_size is not None
        assert max_word_segments is not None
        assert emb_size is not None

        pad = tf.zeros(shape=(1,emb_size), name="padding_vector", dtype=tf.float32)

        segment_in_matr = tf.get_variable("SEGM_IN", shape=(segment_vocab_size - 1, emb_size), dtype=tf.float32)

        in_embedding_matrix = tf.concat([in_matr, segment_in_matr, pad], axis=0)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,max_word_segments), name="in_words")

        in_emb = tf.reduce_mean(tf.nn.embedding_lookup(in_embedding_matrix, in_words), axis=1)

    elif model == "attentive":

        assert vocab_size is not None
        assert segment_vocab_size is not None
        assert max_word_segments is not None
        assert emb_size is not None

        attentive_seq_len = max_word_segments

        # pad = tf.zeros(shape=(1, emb_size), name="padding_vector", dtype=tf.float32)

        segment_in_matr = tf.get_variable("SEGM_IN", shape=(segment_vocab_size, emb_size), dtype=tf.float32)

        in_embedding_matrix = tf.concat([in_matr, segment_in_matr], axis=0)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None, attentive_seq_len), name="in_words")

        emb_segments_in = tf.nn.embedding_lookup(in_embedding_matrix, in_words)

        emb_segments_in_r = tf.reshape(emb_segments_in, (-1, attentive_seq_len * emb_size))

        def attention_layer(input_):
            d_out = tf.nn.dropout(input_, keep_prob=dropout)
            joined_attention = tf.layers.dense(d_out, attentive_seq_len, name='joined_attention', kernel_regularizer=tf.nn.l2_loss)
            attention_mask = tf.reshape(joined_attention, (-1, attentive_seq_len, 1), name='attention_mask')
            soft_attention = tf.nn.softmax(attention_mask, axis=1, name='soft_attention_mask')
            return soft_attention

        with tf.variable_scope('attention') as att_scope:
            emb_segments_in_attention_mask = attention_layer(emb_segments_in_r)

        in_emb = tf.reduce_sum(emb_segments_in * emb_segments_in_attention_mask, axis=1)

    else:
        raise NotImplementedError("Invalid model name: %s" % model)

    final = tf.nn.l2_normalize(in_emb, axis=1)

    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product")
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_mean(per_item_loss, axis=0)

    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

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

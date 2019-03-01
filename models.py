import tensorflow as tf


def assemble_graph(model='skipgram',
                   vocab_size=None,
                   emb_size=None,
                   segment_vocab_size=None,
                   max_word_segments=None):

    counter = tf.Variable(0, dtype=tf.int32)
    adder = tf.assign(counter, counter + 1)

    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learn_rate')
    labels = tf.placeholder(dtype=tf.float32, shape=(None,), name="labels")

    if model == 'skipgram':

        assert vocab_size is not None
        assert emb_size is not None

        # embedding matrices
        in_matr = tf.get_variable("IN", shape=(vocab_size, emb_size), dtype=tf.float32)
        out_matr = tf.get_variable("OUT", shape=(vocab_size, emb_size), dtype=tf.float32)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="in_words")
        out_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="out_words")

        in_emb = tf.nn.embedding_lookup(in_matr, in_words)
        out_emb = tf.nn.embedding_lookup(out_matr, out_words)

    elif model == 'fasttext' or model == 'morph':

        assert segment_vocab_size is not None
        assert max_word_segments is not None
        assert emb_size is not None

        in_matr = tf.get_variable("IN", shape=(segment_vocab_size, emb_size), dtype=tf.float32)
        out_matr = tf.get_variable("OUT", shape=(segment_vocab_size, emb_size), dtype=tf.float32)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,max_word_segments), name="in_words")
        out_words = tf.placeholder(dtype=tf.int32, shape=(None,max_word_segments), name="out_words")

        in_emb = tf.reduce_sum(tf.nn.embedding_lookup(in_matr, in_words), axis=1)
        out_emb = tf.reduce_sum(tf.nn.embedding_lookup(out_matr, out_words), axis=1)

    elif model == "attentive":

        assert segment_vocab_size is not None
        assert max_word_segments is not None
        assert emb_size is not None

        in_matr = tf.get_variable("IN", shape=(segment_vocab_size, emb_size), dtype=tf.float32)
        out_matr = tf.get_variable("OUT", shape=(segment_vocab_size, emb_size), dtype=tf.float32)

        in_words = tf.placeholder(dtype=tf.int32, shape=(None, max_word_segments), name="in_words")
        out_words = tf.placeholder(dtype=tf.int32, shape=(None, max_word_segments), name="out_words")

        emb_segments_in = tf.nn.embedding_lookup(in_matr, in_words)
        emb_segments_out = tf.nn.embedding_lookup(out_matr, out_words)

        emb_segments_in_r = tf.reshape(emb_segments_in, (-1, max_word_segments * emb_size))
        emb_segments_out_r = tf.reshape(emb_segments_out, (-1, max_word_segments * emb_size))

        def attention_layer(input_):
            d_out = tf.nn.dropout(input_, keep_prob=0.7)
            joined_attention = tf.layers.dense(d_out, max_word_segments * emb_size, name='joined_attention')
            attention_mask = tf.reshape(joined_attention, (-1, max_word_segments, emb_size), name='attention_mask')
            soft_attention = tf.nn.softmax(attention_mask, axis=1, name='soft_attention_mask')
            return soft_attention

        with tf.variable_scope('attention') as att_scope:
            emb_segments_in_attention_mask = attention_layer(emb_segments_in_r)
            att_scope.reuse_variables()
            emb_segments_out_attention_mask = attention_layer(emb_segments_out_r)

        in_emb = tf.reduce_sum(emb_segments_in * emb_segments_in_attention_mask, axis=1)
        out_emb = tf.reduce_sum(emb_segments_out * emb_segments_out_attention_mask, axis=1)

    else:
        raise NotImplementedError("Invalid model name: %s" % model)


    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product")
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_mean(per_item_loss, axis=0)

    # train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

    in_out = tf.nn.l2_normalize(0.5 * (in_emb + out_emb), axis=1)

    return {
        'in_words': in_words,
        'out_words': out_words,
        'labels': labels,
        'loss': loss,
        'train': train,
        'adder': adder,
        'in_out': in_out,
        'learning_rate': learning_rate
    }


def embedder(vocab_size,
             emb_size):
    with tf.variable_scope('embeddings_%d' % vocab_size) as scope:
        final = tf.get_variable("final_%d" % vocab_size, shape=(vocab_size, emb_size))

        embs = tf.placeholder(shape=(vocab_size, emb_size), name="vocab_%d_pl" % vocab_size, dtype=tf.float32)
        ids = tf.placeholder(shape=(None, ), name="lookup_ids_%d_pl" % vocab_size, dtype=tf.int32)

        assign = tf.assign(final, embs)

        lookup = tf.nn.embedding_lookup(final, ids, name="lookup_%d_pl" % vocab_size)

    return {
        'assign': assign,
        'embs': embs,
        'lookup': lookup,
        'ids': ids,
        'final': final
    }


def assign_embeddings(sess, summary_writer, in_out_tensor, in_tensor, out_tensor, vocab_size, embedders):
    print("\t\tAssigning vocabulary of size %d" % vocab_size)
    ids = np.array(list(range(vocab_size)))
    if model_name != 'skipgram':
        ids_expanded = segmenter.segment(ids)
    else:
        ids_expanded = ids
    in_out = sess.run(in_out_tensor, {in_tensor: ids_expanded, out_tensor: ids_expanded})
    sess.run(embedders[vocab_size]['assign'], {embedders[vocab_size]['embs']: in_out})

    embs = sess.run(embedders[vocab_size]['lookup'], {embedders[vocab_size]['ids']: ids})

    save_path = saver.save(sess, ckpt_path)

    dump_path = "./embeddings/%s_%d.pkl" % (model_name, vocab_size)
    pickle.dump(embs, open(dump_path, "wb"))

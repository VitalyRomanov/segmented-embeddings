import tensorflow as tf
import sys
from Vocabulary import Vocabulary
from Reader import Reader
from WordSegmenter import WordSegmenter
import pickle
import argparse
import numpy as np
import time
import resource

resource.setrlimit(resource.RLIMIT_AS, (2**33, 2**33))

sys.stdin.readline()
sys.stdin.readline()
n_param = int(sys.stdin.readline())

args = dict([tuple(sys.stdin.readline().strip().split("=")) for _ in range(n_param)])

n_dims = int(args['dimensionality'])
epochs = int(args['epochs'])
n_contexts = int(args['context'])
k = int(args['negative'])
window_size = int(args['window_size'])
model_name = args['model_name']
data_path = args['data_path']
vocabulary_path = args['voc_path']
wiki = bool(args['wiki'])
lang = args['language']
sgm_path = args['segmenter']
full_voc_size = int(args['full_vocabulary_size'])
batch_size = int(args['batch_size'])
graph_saving_path = args['graph_saving_path']
ckpt_path = args['ckpt_path']


vocab_progressions = [10000, 20000, 50000, 100000, 200000]


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
        'learning_rate': learning_rate,
        'batch_count': counter
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


def assign_embeddings(sess, in_out_tensor, in_tensor, out_tensor, vocab_size, embedders):
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


print("Starting training", time.asctime( time.localtime(time.time()) ))

if model_name != 'skipgram':
    segmenter = WordSegmenter(sgm_path, lang)
    sgm = segmenter.segment

    segm_voc_size = segmenter.unique_segments
    word_segments = segmenter.max_len

    terminals = assemble_graph(model=model_name,
                               segment_vocab_size=segm_voc_size,
                               max_word_segments=word_segments,
                               emb_size=n_dims)
else:

    terminals = assemble_graph(model=model_name,
                               vocab_size=full_voc_size,
                               emb_size=n_dims)


first_batch = None

in_words_ = terminals['in_words']
out_words_ = terminals['out_words']
labels_ = terminals['labels']
train_ = terminals['train']
loss_ = terminals['loss']
adder_ = terminals['adder']
in_out_ = terminals['in_out']
lr_ = terminals['learning_rate']
batch_count_ = terminals['batch_count']


saver = tf.train.Saver()
saveloss_ = tf.summary.scalar('loss', loss_)


embedders = {voc_size: embedder(voc_size, n_dims) for voc_size in vocab_progressions}


def parse_model_input(model_name, line):

    try:
        a_, p_, l_ = line.split("\t")
    except:
        print(line)
        return -1, -1, -1, False

    a = int(a_)
    p = int(p_)
    l = int(l_)
    return a, p, l, True

    # if model_name != 'skipgram':
    #
    #     a = list(map(int, a_.split()))
    #     p = list(map(int, p_.split()))
    #     l = [int(l_)]
    #     return a, p, l, True
    #
    # else:
    #     a_, p_, l_ = line.split("\t")
    #     a = [int(a_)]
    #     p = [int(p_)]
    #     l = [int(l_)]
    #     return a, p, l, True


in_batch = []
out_batch = []
lbl_batch = []


def flush():
    global in_batch, out_batch, lbl_batch
    in_batch = []
    out_batch = []
    lbl_batch = []


def save_snapshot(sess, terminals, vocab_size, embedders):
    batch_count = sess.run(terminals['batch_count'])
    assign_embeddings(sess, terminals['in_out'],
                      terminals['in_words'],
                      terminals['out_words'],
                      vocab_size,
                      embedders)
    path = "./models/%s_%d_%d" % (model_name, vocab_size, batch_count)
    ckpt_p = "%s/model.ckpt" % path
    save_path = saver.save(sess, ckpt_p)


def create_batch(model_name, in_batch, out_batch, lbl_batch):
    if model_name != 'skipgram':
        return sgm(np.array(in_batch)), sgm(np.array(out_batch)), np.float32(np.array(lbl_batch))
    else:
        return np.array(in_batch), np.array(out_batch), np.float32(np.array(lbl_batch))

vocab_size = 10000
epoch = 0

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(graph_saving_path, graph=sess.graph)

    # Restore from checkpoint
    # saver.restore(sess, ckpt_path)
    # sess.graph.as_default()

    for line in iter(sys.stdin.readline, ""):

        training_stages = line.strip().split("=")

        if len(training_stages) > 1:
            if training_stages[0] == 'vocab':

                new_vocab_size = int(training_stages[1])

                if new_vocab_size != vocab_size:
                    epoch = 0
                    flush()

                    save_snapshot(sess, terminals, vocab_size, embedders)

                    epochs += 2

                    vocab_size = new_vocab_size

            elif training_stages[0] == 'epoch':
                epoch = int(training_stages[1])
                flush()
            else:
                raise Exception("Unknown sequence: %s" % line.strip())

        learn_rate = 0.025 * (1. - epoch / epochs)

        in_, out_, lbl_, valid = parse_model_input(model_name, line.strip())

        if valid:
            in_batch.append(in_)
            out_batch.append(out_)
            lbl_batch.append(lbl_)

            if len(in_batch) == batch_size:

                in_b, out_b, lbl_b = create_batch(model_name, in_batch, out_batch, lbl_batch)

                _, batch_count = sess.run([train_, adder_], {
                    in_words_: in_b,
                    out_words_: out_b,
                    labels_: lbl_b,
                    lr_: learn_rate
                })

                if batch_count % 10 == 0:
                    # in_words, out_words, labels = first_batch
                    loss_val, summary = sess.run([loss_, saveloss_], {
                        in_words_: in_b,
                        out_words_: out_b,
                        labels_: lbl_b
                    })
                    print("\t\tVocab: {}, Epoch {}, batch {}, loss {}".format(vocab_size, epoch, batch_count, loss_val))
                    save_path = saver.save(sess, ckpt_path)
                    summary_writer.add_summary(summary, batch_count)

    save_snapshot(sess, terminals, vocab_size, embedders)

print("Finished trainig", time.asctime( time.localtime(time.time()) ))
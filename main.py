import tensorflow as tf
import sys
from Vocabulary import Vocabulary
from Reader import Reader
from WordSegmenter import WordSegmenter
import pickle
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('-d', type=int, default=50, dest='dimensionality', help='Trained embedding dimensionality')
parser.add_argument('-e', type=int, default=5, dest='epochs', help='Trained embedding dimensionality')
parser.add_argument('-c', type=int, default=20, dest='context', help='Number of contexts in batch')
parser.add_argument('-n', type=int, default=5, dest='negative', help='Number of negative samples')
parser.add_argument('-w', type=int, default=5, dest='window_size', help='Context window size (on one side)')
parser.add_argument('-m', type=str, default='skipgram', dest='model_name', help='Trained model')
parser.add_argument('-s', type=float, default=1e-4, dest='subsampling_parameter', help='Subsampling threshold')
parser.add_argument('-l', type=str, default='en', dest='language', help='Language of wikipedia dump')
parser.add_argument('-sgm', type=str, dest='segmenter', help='Segmenter Path')
parser.add_argument('-wiki', type=bool, default=False, dest='wiki', help='Read from wikipedia dump')
parser.add_argument('data_path', type=str, help='Path to training data. Can be plain file or wikipedia dump. Set flag \'--wiki\' if using wiki dump')
parser.add_argument('voc_path', type=str, help='Path to vocabulary dump')


args = parser.parse_args()
n_dims = args.dimensionality
epochs = args.epochs
n_contexts = args.context
k = args.negative
window_size = args.window_size
model_name = args.model_name
data_path = args.data_path
vocabulary_path = args.voc_path
wiki = args.wiki
lang = args.language
sgm_path = args.segmenter

graph_saving_path = "./models/%s" % model_name
ckpt_path = "%s/model.ckpt" % graph_saving_path
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

        in_emb = tf.reduce_sum(tf.nn.embedding_lookup(in_matr, in_words), axis=2)
        out_emb = tf.reduce_sum(tf.nn.embedding_lookup(out_matr, out_words), axis=2)

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
            joined_attention = tf.layers.dense(input_, max_word_segments * emb_size, name='joined_attention')
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

    in_out = tf.nn.l2_normalize(0.5 * (in_emb + out_emb))

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


def embedder(in_out_tensor,
             vocab_size,
             emb_size):
    with tf.variable_scope('embeddings_%d' % vocab_size) as scope:
        final = tf.get_variable("final_%d" % vocab_size, shape=(vocab_size, emb_size))

        embs = tf.placeholder(shape=(vocab_size, emb_size), name="vocab_%d_pl" % vocab_size, dtype=tf.float32)

        assign = tf.assign(final, embs)

    return {
        'assign': assign,
    }


import time
print("Reading data", time.asctime( time.localtime(time.time()) ))

# Estimate vocabulary from training data
voc = pickle.load(open(vocabulary_path, "rb"))
# voc = Vocabulary()
# with open(data_path, "r") as data:
#     line = data.readline()
#     while line:
#         tokens = line.strip().split()
#         voc.add_words(tokens)
#         line = data.readline()
# voc.prune(top_words)
# voc.export_vocabulary(top_words, "voc.tsv")
# voc.save("voc.pkl")

voc.set_subsampling_param(1e-4)

print("Starting training", time.asctime( time.localtime(time.time()) ))

reader = Reader(data_path, voc, n_contexts, window_size, k, wiki=wiki, lang=lang)


if model_name != 'skipgram':
    segmenter = WordSegmenter(sgm_path, lang)
    sgm = segmenter.segment

    def next_batch(from_top_n=None):
        pos, neg, lbl = reader.next_batch(from_top_n=from_top_n)
        return sgm(pos), sgm(neg), lbl

    restore = segmenter.to_segments

    segm_voc_size = segmenter.unique_segments
    word_segments = segmenter.max_len

    terminals = assemble_graph(model=model_name,
                               segment_vocab_size=segm_voc_size,
                               max_word_segments=word_segments,
                               emb_size=n_dims)
else:
    def next_batch(from_top_n=None):
        return reader.next_batch(from_top_n=from_top_n)


    restore = lambda x: voc.id2word[x]

    terminals = assemble_graph(model=model_name,
                               vocab_size=len(voc),
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

saver = tf.train.Saver()
saveloss_ = tf.summary.scalar('loss', loss_)

# batch = next_batch(from_top_n=vocab_progressions[0])
# while batch is not None:
#     for a, p, l in zip(batch[0].tolist(), batch[1].tolist(), batch[2].tolist()):
#         print(restore(a), restore(p), l)
#     batch = next_batch(from_top_n=vocab_progressions[0])
#     sys.exit()
# # print(time.asctime( time.localtime(time.time()) ))
#
# batch = next_batch(from_top_n=vocab_progressions[0])
# for a, p, l in zip(batch[0].tolist(), batch[1].tolist(), batch[2].tolist()):
#     print(restore(a), restore(p), l)
# print(time.asctime( time.localtime(time.time()) ))



embedders = {voc_size: embedder(in_out_, voc_size, n_dims) for voc_size in vocab_progressions}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(graph_saving_path, graph=sess.graph)

    sys.exit()

    # Restore from checkpoint
    # saver.restore(sess, ckpt_path)
    # sess.graph.as_default()

    for vocab_size in vocab_progressions:

        for e in range(epochs):
            batch = next_batch(from_top_n=vocab_size)
            first_batch = batch

            learn_rate = 0.025 * (1. - e/epochs)

            while batch is not None:

                in_words, out_words, labels = batch

                _, batch_count = sess.run([train_, adder_], {
                    in_words_: in_words,
                    out_words_: out_words,
                    labels_: labels,
                    lr_: learn_rate
                })

                if batch_count % 1000 == 0:
                    # in_words, out_words, labels = first_batch
                    loss_val, summary= sess.run([loss_, saveloss_], {
                        in_words_: in_words,
                        out_words_: out_words,
                        labels_: labels
                    })
                    print("\t\tVocab: {}, Epoch {}, batch {}, loss {}".format(vocab_size, e, batch_count, loss_val))
                    save_path = saver.save(sess, ckpt_path)
                    summary_writer.add_summary(summary, batch_count)

                batch = next_batch(from_top_n=vocab_size)

        save_path = saver.save(sess, graph_saving_path+"_"+str(vocab_size)+"_"+str(batch_count))
        epochs += 2

print("Finished trainig", time.asctime( time.localtime(time.time()) ))
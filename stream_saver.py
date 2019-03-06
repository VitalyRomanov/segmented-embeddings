import tensorflow as tf
import sys
from Vocabulary import Vocabulary
from Reader import Reader
from WordSegmenter import WordSegmenter
import pickle
import argparse
import numpy as np
import time

from models import assemble_graph


sys.stdin.readline()
sys.stdin.readline()
n_param = int(sys.stdin.readline())

args = dict([tuple(sys.stdin.readline().strip().split("=")) for _ in range(n_param)])

n_dims = int(args['dimensionality'])
model_name = args['model_name']
lang = args['language']
sgm_path = args['segmenter']
sgm_len = int(args['segmenter_len'])
full_voc_size = int(args['full_vocabulary_size'])
graph_saving_path = args['graph_saving_path']
ckpt_path = args['ckpt_path']
vocab_size = int(args['vocabulary_size'])


restore = 1
if restore:
    print("Restoring from checkpoint")
else:
    print("Training from scratch")



def assign_embeddings(sess, terminals, vocab_size):
    in_words_ = terminals['in_words']
    final_ = terminals['final']
    dropout_ = terminals['dropout']
    attention_ = terminals['attention_mask']

    print("\t\tDumpung vocabulary of size %d" % vocab_size)
    ids = np.array(list(range(vocab_size)))
    if model_name != 'skipgram':
        ids_expanded = segmenter.segment(ids)
    else:
        ids_expanded = ids

    final = sess.run(final_, {in_words_: ids_expanded,
                              dropout_: 1.0})

    emb_dump_path = "./embeddings/%s_%d.pkl" % (model_name, vocab_size)

    if model_name == 'attentive':
        sgm_p = sgm_path.split("/")[0]
        emb_dump_path = "./embeddings/%s_%s_%d.pkl" % (model_name, sgm_p, vocab_size)
        dump_path = "./embeddings/attention_mask_%s_%s_%d.pkl" % (sgm_p, model_name, vocab_size)

        attention_mask = sess.run(attention_, {in_words_: ids_expanded,
                              dropout_: 1.0})
        pickle.dump(attention_mask, open(dump_path, "wb"))

    pickle.dump(final, open(emb_dump_path, "wb"))



print("Starting saving", time.asctime( time.localtime(time.time()) ))

if model_name != 'skipgram':
    segmenter = WordSegmenter(sgm_path, lang, sgm_len)
    sgm = segmenter.segment

    segm_voc_size = segmenter.unique_segments
    word_segments = segmenter.max_len

    print("Max Word Len is %d segments" % word_segments)

    terminals = assemble_graph(model=model_name,
                               vocab_size=full_voc_size,
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
lr_ = terminals['learning_rate']
batch_count_ = terminals['batch_count']


saver = tf.train.Saver()
saveloss_ = tf.summary.scalar('loss', loss_)


def save_snapshot(sess, terminals, vocab_size):
    batch_count = sess.run(terminals['batch_count'])
    path = "./models/%s_%d_%d" % (model_name, vocab_size, batch_count)
    ckpt_p = "%s/model.ckpt" % path
    assign_embeddings(sess, terminals, vocab_size)
    save_path = saver.save(sess, ckpt_p)


epoch = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(graph_saving_path, graph=sess.graph)

    # Restore from checkpoint
    if restore:
        saver.restore(sess, ckpt_path)
        sess.graph.as_default()

    save_snapshot(sess, terminals, vocab_size)

print("Finished saving", time.asctime( time.localtime(time.time()) ))
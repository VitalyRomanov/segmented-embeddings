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
restore = int(args['restore'])
gpu_mem = args['gpu_mem']


if restore:
    print("Restoring from checkpoint")
else:
    print("Training from scratch")



def assign_embeddings(sess, terminals, vocab_size):
    in_words_ = terminals['in_words']
    final_ = terminals['final']
    dropout_ = terminals['dropout']

    print("\t\tDumpung vocabulary of size %d" % vocab_size)
    ids = np.array(list(range(vocab_size)))
    if model_name != 'skipgram':
        ids_expanded = segmenter.segment(ids)
    else:
        ids_expanded = ids

    final = sess.run(final_, {in_words_: ids_expanded,
                              dropout_: 1.0})

    dump_path = "./embeddings/%s_%d.pkl" % (model_name, vocab_size)
    pickle.dump(final, open(dump_path, "wb"))


print("Starting training", time.asctime( time.localtime(time.time()) ))

if model_name != 'skipgram':
    segmenter = WordSegmenter(sgm_path, lang)
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
dropout_ = terminals['dropout']


saver = tf.train.Saver()
saveloss_ = tf.summary.scalar('loss', loss_)


def parse_model_input(line):

    try:
        a_, p_, l_ = line.split("\t")
    except:
        print(line)
        return -1, -1, -1, False

    a = int(a_)
    p = int(p_)
    l = int(l_)
    return a, p, l, True


in_batch = []
out_batch = []
lbl_batch = []


def flush():
    global in_batch, out_batch, lbl_batch
    in_batch = []
    out_batch = []
    lbl_batch = []


def save_snapshot(sess, terminals, vocab_size):
    batch_count = sess.run(terminals['batch_count'])
    path = "./models/%s_%d_%d" % (model_name, vocab_size, batch_count)
    ckpt_p = "%s/model.ckpt" % path
    assign_embeddings(sess, terminals, vocab_size)
    save_path = saver.save(sess, ckpt_p)


def create_batch(model_name, in_batch, out_batch, lbl_batch):
    if model_name != 'skipgram':
        if model_name == "attentive":
            return sgm(np.array(in_batch))[:,:-1] - segmenter.total_words, np.array(out_batch), np.float32(np.array(lbl_batch))
        else:
            return sgm(np.array(in_batch)), np.array(out_batch), np.float32(np.array(lbl_batch))
    else:
        return np.array(in_batch), np.array(out_batch), np.float32(np.array(lbl_batch))

vocab_size = 100000
epoch = 0
initial_learn_rate = 0.05
learn_rate = initial_learn_rate


save_every = 2000 * 50000 // batch_size


if gpu_mem == 'None':
    gpu_options = tf.GPUOptions()
else:
    frac = float(gpu_mem)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(graph_saving_path, graph=sess.graph)

    # Restore from checkpoint
    if restore:
        saver.restore(sess, ckpt_path)
        sess.graph.as_default()

    for line in iter(sys.stdin.readline, ""):

        training_stages = line.strip().split("=")

        if len(training_stages) > 1:
            if training_stages[0] == 'vocab':

                new_vocab_size = int(training_stages[1])

                if new_vocab_size != vocab_size:
                    epoch = 0
                    flush()

                    save_snapshot(sess, terminals, vocab_size)

                    epochs += 0

                    vocab_size = new_vocab_size

            elif training_stages[0] == 'epoch':
                epoch = int(training_stages[1])
                flush()
            else:
                raise Exception("Unknown sequence: %s" % line.strip())



        in_, out_, lbl_, valid = parse_model_input(line.strip())

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
                    lr_: learn_rate,
                    dropout_: 0.7
                })

                learn_rate = initial_learn_rate * (1. - batch_count / 10000000)

                if batch_count % save_every == 0:
                    # in_words, out_words, labels = first_batch
                    loss_val, summary = sess.run([loss_, saveloss_], {
                        in_words_: in_b,
                        out_words_: out_b,
                        labels_: lbl_b,
                        dropout_: 1.0
                    })
                    print("\t\tVocab: {}, Epoch {}, batch {}, loss {}".format(vocab_size, epoch, batch_count, loss_val))
                    save_path = saver.save(sess, ckpt_path)
                    summary_writer.add_summary(summary, batch_count)

                flush()

    save_snapshot(sess, terminals, vocab_size)

print("Finished trainig", time.asctime( time.localtime(time.time()) ))
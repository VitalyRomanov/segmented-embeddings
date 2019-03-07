import tensorflow as tf
import sys
from WordSegmenter import WordSegmenter
import pickle
import numpy as np
import time
from ast import literal_eval

from models import assemble_graph


sys.stdin.readline()
sys.stdin.readline()
n_param = int(sys.stdin.readline())

args = dict([tuple(sys.stdin.readline().strip().split("=")) for _ in range(n_param)])

n_dims = int(args['dimensionality'])
epochs = int(args['epochs'])
# n_contexts = int(args['context'])
# k = int(args['negative'])
# window_size = int(args['window_size'])
model_name = args['model_name']
data_path = args['data_path']
# vocabulary_path = args['voc_path']
# wiki = bool(args['wiki'])
# lang = args['language']
# sgm_path = args['segmenter']
sgm_len = int(args['segmenter_len'])
sgm_voc_size = int(args['segm_voc_size'])
full_voc_size = int(args['full_vocabulary_size'])
# batch_size = int(args['batch_size'])
graph_saving_path = args['graph_saving_path']
ckpt_path = args['ckpt_path']
restore = int(args['restore'])
gpu_mem = args['gpu_mem']
vocab_size = int(args['vocabulary_size'])


for key,val in args.items():
    print("{}={}".format(key, val))

print()

if restore:
    print("Restoring from checkpoint\n")
else:
    print("Training from scratch\n")



def assign_embeddings(sess, terminals, vocab_size):
    in_words_ = terminals['in_words']
    final_ = terminals['final']

    print("\t\tDumpung vocabulary of size %d" % vocab_size)
    ids = np.array(list(range(vocab_size)))

    final = sess.run(final_, {in_words_: ids})

    emb_dump_path = "./embeddings/%s_%d.pkl" % (model_name, vocab_size)

    pickle.dump(final, open(emb_dump_path, "wb"))



print("Assembling model", time.asctime( time.localtime(time.time()) ))

terminals = assemble_graph(model=model_name,
                           vocab_size=full_voc_size,
                           segment_vocab_size=sgm_voc_size,
                           emb_size=n_dims)


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


def parse_model_input(line):
    batch = pickle.loads(literal_eval(line))

    return batch[:, 0], batch[:, 1], batch[:, 2]


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


epoch = 0
init_learn_rate = 0.05
learn_rate = init_learn_rate
wiki_step = 0
wiki_ceil = 6000


print("Starting training", time.asctime( time.localtime(time.time()) ))

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

        if len(training_stages) == 2:
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
            print(line.strip())
            continue

        if len(line) > 4 and line[:4] == 'wiki':
            wiki_step += 1
            learn_rate = init_learn_rate * (1 - wiki_step / wiki_ceil)
            print("%s learning_rate=%.4f" % (line.strip(), learn_rate))

            if line.strip() == "wiki_99":
                loss_val, summary = sess.run([loss_, saveloss_], feed_dict = {
                    in_words_: in_b,
                    out_words_: out_b,
                    labels_: lbl_b,
                })
                print("\t\tVocab: {}, Epoch {}, batch {}, loss {}".format(vocab_size, epoch, batch_count, loss_val))
                save_path = saver.save(sess, ckpt_path)
                summary_writer.add_summary(summary, batch_count)

            continue

        if not line.strip(): continue

        in_b, out_b, lbl_b = parse_model_input(line.strip())

        _, batch_count = sess.run([train_, adder_], {
            in_words_: in_b,
            out_words_: out_b,
            labels_: lbl_b,
            lr_: learn_rate
        })

    save_snapshot(sess, terminals, vocab_size)

print("Finished trainig", time.asctime( time.localtime(time.time()) ))
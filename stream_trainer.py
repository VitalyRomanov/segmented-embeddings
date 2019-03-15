import tensorflow as tf
import sys

import numpy as np
import time
import pickle
from ast import literal_eval
from aux import format_args, get_model, save_snapshot



sys.stdin.readline()
sys.stdin.readline()
n_param = int(sys.stdin.readline())
args = dict([tuple(sys.stdin.readline().strip().split("=")) for _ in range(n_param)])
args = format_args(args)


for key,val in args.items():
    print("{}={}".format(key, val))


if args['restore']:
    print("Restoring from checkpoint\n")
else:
    print("Training from scratch\n")


print("Assembling model", time.asctime( time.localtime(time.time()) ))

model = get_model(args)


def parse_model_input(line):

    try:
        batch = pickle.loads(literal_eval(line))
    except:
        global wiki_step, learn_rate, init_learn_rate, processed_tokens

        try:
            w = line.split("\t")[0][:4]
            if w == 'wiki':
                wiki_step += 1
                learn_rate = init_learn_rate * (1 - wiki_step / wiki_ceil)
        finally:
            print("%s learning_rate=%.4f processed_tokens=%d" % (line, learn_rate, processed_tokens))
            return None

    return batch


def flush():
    global r_batches, batch_size
    r_batches = []
    batch_size = 0


# def create_batch(model_name, r_batches):
#     batch = np.vstack(r_batches)
#
#     if model_name != 'skipgram':
#         return sgm(batch[:, 0]), batch[:, 1], np.float32(batch[:, 2])
#     else:
#         return batch[:, 0], batch[:, 1], np.float32(batch[:, 2])


epoch = 0
init_learn_rate = args['learning_rate']
learn_rate = init_learn_rate
wiki_step = 0
wiki_ceil = 101


save_every = 1000000
processed_tokens = 0
batch_size = 0
r_batches = []

print("Starting training", time.asctime( time.localtime(time.time()) ))

if args['restore']:
    model.restore_graph()

for line in iter(sys.stdin.readline, ""):

    batch = parse_model_input(line.strip())

    if learn_rate < 0:
        print("Learning rate is suddenly negative")
        model.save_snapshot()
        break

    if batch is not None:
        r_batches.append(batch)
        batch_size += batch.shape[0]
        processed_tokens += args['context']

        if batch_size >= args['batch_size']:
            s_batch = np.vstack(r_batches)

            model.update(s_batch, lr=learn_rate)

            flush()

        if processed_tokens % save_every == 0:

            loss_val, batch_count = model.evaluate(s_batch, save=True)

            print("Vocab: {}, Epoch {}, batch {}, loss {}".format(args['vocabulary_size'],
                                                                  epoch,
                                                                  batch_count,
                                                                  loss_val))

model.save_snapshot()

# # if args['gpu_mem'] == 0.:
# #     gpu_options = tf.GPUOptions()
# # else:
# #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args['gpu_mem'])
# #
# # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# # with tf.Session() as sess:
#
#     # summary_writer = tf.summary.FileWriter(args['graph_path'],
#     #                                        graph=sess.graph)
#
#     # Restore from checkpoint
#     # if args['restore']:
#         # try:
#         #     saver.restore(sess, args['ckpt_path'])
#         #     sess.graph.as_default()
#         # except:
#         #     print("Cannot restore: checkpoint does not exist")
#         #     sys.exit()
#
#     for line in iter(sys.stdin.readline, ""):
#
#         batch = parse_model_input(line.strip())
#
#         if learn_rate < 0:
#             print("Learning rate is suddenly negative")
#             save_snapshot(sess, saver, terminals, args)
#             break
#
#         if batch is not None:
#             r_batches.append(batch)
#             batch_size += batch.shape[0]
#             processed_tokens += args['context']
#
#             if batch_size >= args['batch_size']:
#
#                 in_b, out_b, lbl_b = create_batch(args['model_name'], r_batches)
#
#                 _, batch_count = sess.run([train_, adder_], feed_dict = {
#                     in_words_: in_b,
#                     out_words_: out_b,
#                     labels_: lbl_b,
#                     lr_: learn_rate,
#                     dropout_: 0.7
#                 })
#
#                 flush()
#
#             if processed_tokens % save_every == 0:
#
#                 loss_val, summary = sess.run([loss_, saveloss_], feed_dict = {
#                     in_words_: in_b,
#                     out_words_: out_b,
#                     labels_: lbl_b,
#                     lr_: learn_rate,
#                     dropout_: 1.0
#                 })
#
#                 print("Vocab: {}, Epoch {}, batch {}, loss {}".format(args['vocabulary_size'],
#                                                                       epoch,
#                                                                       batch_count,
#                                                                       loss_val))
#                 # save_path = saver.save(sess, ckpt_path)
#                 summary_writer.add_summary(summary, batch_count)
#
#
#     save_snapshot(sess, saver, terminals, args)

print("Finished trainig", time.asctime( time.localtime(time.time()) ))

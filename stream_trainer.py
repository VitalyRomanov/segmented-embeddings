# import tensorflow as tf
import sys

import numpy as np
import time
import pickle
from ast import literal_eval
from aux import format_args, get_model


sys.stdin.readline()
sys.stdin.readline()
n_param = int(sys.stdin.readline())
args = dict([tuple(sys.stdin.readline().strip().split("=")) for _ in range(n_param)])
args = format_args(args)


for key,val in args.items():
    print("{}={}".format(key, val))

if args['save']:
    print("Saving from checkpoint\n")
elif args['restore']:
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


epoch = 0
init_learn_rate = args['learning_rate']
learn_rate = init_learn_rate
wiki_step = 0
wiki_ceil = args['learning_rate_decay']


save_every = 1000000
processed_tokens = 0
batch_size = 0
r_batches = []

print("Starting training", time.asctime( time.localtime(time.time()) ))

if args['restore']:# or args['save']:
    model.restore_graph()

if not args['save']:

    for line in iter(sys.stdin.readline, ""):

        batch = parse_model_input(line.strip())

        if learn_rate < 0:
            print("Learning rate is suddenly negative")
            model.save_snapshot()
            break

        if batch is not None:
            r_batches.append(batch)
            batch_size += args['context'] #batch.shape[0]
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

print("Finished trainig", time.asctime( time.localtime(time.time()) ))

import tensorflow as tf
import numpy as np
import sys
import pickle


class Skipgram:
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None,
                 model_name='skipgram'):

        self.graph_path = graph_path
        self.vocabulary_size = vocab_size
        self.emb_size = emb_size
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.assemble_model()

        self.open_session(gpu_options)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.graph_path,
                                               graph=self.sess.graph)

    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()

    def assemble_model(self):
        counter = tf.Variable(0, dtype=tf.int32)
        adder = tf.assign(counter, counter + 1)

        learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                       name='learn_rate')

        labels = tf.placeholder(dtype=tf.float32, shape=(None,),
                                name="labels")

        in_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                  name="in_words")

        out_words = tf.placeholder(dtype=tf.int32, shape=(None,),
                                   name="out_words")

        in_matr = tf.get_variable("IN", shape=[self.vocabulary_size, self.emb_size],
                                  dtype=tf.float32)

        out_matr = tf.get_variable("OUT", shape=[self.vocabulary_size, self.emb_size],
                                   dtype=tf.float32)

        out_bias = tf.get_variable("out_bias", dtype=tf.float32,
                                   initializer=tf.zeros([self.vocabulary_size]))

        in_emb = tf.nn.embedding_lookup(in_matr, in_words, name="in_lookup")

        out_emb = tf.nn.embedding_lookup(out_matr, out_words,
                                         name="out_lookup")
        bias_slice = tf.gather_nd(out_bias, tf.reshape(out_words, (-1, 1)))

        logits = tf.reduce_sum(in_emb * out_emb,
                               axis=1, name="inner_product") + bias_slice

        per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        loss = tf.reduce_mean(per_item_loss, axis=0)

        train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        final = tf.nn.l2_normalize(in_emb, axis=1)

        saveloss = tf.summary.scalar('loss', loss)

        self.terminals = {
            'in_words': in_words,
            'out_words': out_words,
            'labels': labels,
            'loss': loss,
            'train': train,
            'adder': adder,
            'learning_rate': learning_rate,
            'batch_count': counter,
            'final': final,
            'saveloss': saveloss
        }

    def open_session(self, gpu_options):
        if gpu_options:
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def restore_graph(self, path=None):
        try:
            if not path:
                path = self.ckpt_path
            self.saver.restore(self.sess, path)
            self.sess.graph.as_default()
        except:
            print("Cannot restore: checkpoint does not exist")
            sys.exit()

    def save_snapshot(self):
        batch_count = self.sess.run(self.terminals['batch_count'])
        path = "./%s/%s_%d_%d" % (self.graph_path,
                                  self.model_name,
                                  self.vocabulary_size,
                                  batch_count)
        ckpt_p = "%s/model.ckpt" % path

        print("\nDumpung vocabulary of size %d\n" % self.vocabulary_size)
        # ids = np.array(list(range(self.vocabulary_size)))
        # final = self.sess.run(self.terminals['final'], {self.terminals['in_words']: ids})
        final = self.embed_words()

        emb_dump_path = "./embeddings/%s_%d.pkl" % (self.model_name, self.vocabulary_size)
        pickle.dump(final, open(emb_dump_path, "wb"))

        _ = self.saver.save(self.sess, ckpt_p)

    def embed_words(self):
        final = []
        batch_size = 50
        for offset in range(0, self.vocabulary_size, batch_size):
            ids = self.expand_ids(np.array(list(range(offset, min(offset + batch_size, self.vocabulary_size)))))
            embs = self.sess.run(self.terminals['final'], {self.terminals['in_words']: ids})
            final.append(embs)

        return np.vstack(final)

    def expand_ids(self, ids):
        return ids

    def update(self, batch, lr=0.001):
        train_ = self.terminals['train']
        adder_ = self.terminals['adder']
        _, batch_count = self.sess.run([train_, adder_], feed_dict={
            self.terminals['in_words']: batch[:, 0],
            self.terminals['out_words']: batch[:, 1],
            self.terminals['labels']: batch[:, 2],
            self.terminals['learning_rate']: lr,
        })

    def evaluate(self, batch, save=False):
        loss_val, summary, batch_count = self.sess.run([self.terminals['loss'], self.terminals['saveloss'], self.terminals['batch_count']], feed_dict={
            self.terminals['in_words']: batch[:, 0],
            self.terminals['out_words']: batch[:, 1],
            self.terminals['labels']: batch[:, 2],
        })

        self.summary_writer.add_summary(summary, batch_count)

        if save:
            self.saver.save(self.sess, self.ckpt_path)

        return loss_val, batch_count
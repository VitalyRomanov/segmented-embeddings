import argparse
from copy import copy
from models import Skipgram, Fasttext, Morph, MorphGram, SubwordCNN, GPUOptions


# from models import assign_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description='Train word vectors')
    parser.add_argument('-d', type=int, default=300, dest='dimensionality', help='Trained embedding dimensionality')
    parser.add_argument('-e', type=int, default=1, dest='epochs', help='Trained embedding dimensionality')
    parser.add_argument('-c', type=int, default=200, dest='context', help='Number of contexts in batch')
    parser.add_argument('-n', type=int, default=20, dest='negative', help='Number of negative samples')
    parser.add_argument('-w', type=int, default=5, dest='window_size', help='Context window size (on one side)')
    parser.add_argument('-b', type=int, default=1000, dest='batch_size', help='Training batch size')
    parser.add_argument('-v', type=int, default=100000, dest='vocabulary_size', help='Size of vocabulary to train')
    parser.add_argument('-m', type=str, default='skipgram', dest='model_name', help='Trained model')
    parser.add_argument('-s', type=float, default=1e-4, dest='subsampling_parameter', help='Subsampling threshold')
    parser.add_argument('-l', type=str, default='en', dest='language', help='Language of wikipedia dump')
    parser.add_argument('-sgm', type=str, default="", dest='segmenter', help='Segmenter Path')
    parser.add_argument('-sgmlen', type=int, default=0, dest='segmenter_len',
                        help='Maximum length of segmented sequence')
    parser.add_argument('-wiki', action='store_true', help='Read from wikipedia dump')
    parser.add_argument('-restore', action='store_true', help='Restore from checkpoint')
    parser.add_argument('-gm', type=float, default=0.0, dest='gpu_mem', help='Fraction of GPU memory to use')
    parser.add_argument('-lr', type=float, default=0.01, dest='learning_rate', help='Initial learning rate')
    parser.add_argument('-lrdec', type=int, default=101, dest='learning_rate_decay', help='Learning rate decay delay')
    parser.add_argument('data_path', type=str,
                        help='Path to training data. Can be plain file or wikipedia dump. Set flag \'--wiki\' if using wiki dump')
    parser.add_argument('-voc', type=str, default="", dest='voc_path', help='Path to vocabulary dump')
    parser.add_argument('-graph', type=str, default="", dest='graph_path', help='Graph saving path')
    parser.add_argument('-ckpt', type=str, default="", dest='ckpt_path', help='CKPT saving path')
    parser.add_argument('-save', action='store_true', help='Save embeddings')

    args = parser.parse_args()

    if args.segmenter == "":
        if args.model_name == "fasttext":
            args.segmenter = "n_gram_segmentation/" + args.language
            args.segmenter_len = 15
        elif args.model_name == "morph":
            args.segmenter = "morpheme_segmentation/" + args.language
            args.segmenter_len = 8
        elif args.model_name == 'morphgram':
            args.segmenter = "n_gram_segmentation/" + args.language + \
                             "__" + \
                             "morpheme_segmentation/" + args.language
            args.segmenter_len = "15__8"
        elif args.model_name == "subwordcnn":
            args.segmenter = "3_gram_segmentation/" + args.language + \
                             "__" + \
                             "morpheme_segmentation/" + args.language + \
                             "__" + \
                             "lemmas/" + args.language
            args.segmenter_len = "15__8__1"

    if args.voc_path == "":
        # args.voc_path = "vocabularies/%s/%s_voc_tokenized.pkl" % (args.language, args.language)
        args.voc_path = "vocabularies/%s/%s_wc.pkl" % (args.language, args.language)

    if args.graph_path == "":
        args.graph_path = "./output/%s" % args.model_name

    if args.ckpt_path == "":
        args.ckpt_path = "%s/model.ckpt" % args.graph_path

    return args


def format_args(args):
    args = copy(args)

    args['dimensionality'] = int(args['dimensionality'])
    args['epochs'] = int(args['epochs'])
    args['context'] = int(args['context'])
    args['negative'] = int(args['negative'])
    args['window_size'] = int(args['window_size'])
    # model_name = args['model_name']
    # data_path = args['data_path']
    # vocabulary_path = args['voc_path']
    args['wiki'] = True if args['wiki'] == 'True' else False
    # lang = args['language']
    # sgm_path = args['segmenter']
    if args['model_name'] == 'morphgram':
        args['segmenter_len'] = list(map(int, args['segmenter_len'].split("__")))
        args['segmenter'] = args['segmenter'].split("__")
    elif args['model_name'] == 'subwordcnn':
        args['segmenter_len'] = list(map(int, args['segmenter_len'].split("__")))
        args['segmenter'] = args['segmenter'].split("__")
    else:
        args['segmenter_len'] = int(args['segmenter_len'])
    args['batch_size'] = int(args['batch_size'])
    # graph_saving_path = args['graph_path']
    # ckpt_path = args['ckpt_path']
    args['restore'] = True if args['restore'] == 'True' else False
    args['gpu_mem'] = float(args['gpu_mem'])
    args['vocabulary_size'] = int(args['vocabulary_size'])
    args['learning_rate'] = float(args['learning_rate'])
    args['learning_rate_decay'] = int(args['learning_rate_decay'])
    args['save'] = True if args['save'] == 'True' else False
    return args


def get_model(args):
    # from models import assemble_graph

    if args['gpu_mem'] == 0.:
        gpu_options = GPUOptions()
    else:
        gpu_options = GPUOptions(per_process_gpu_memory_fraction=args['gpu_mem'])

    # if args['model_name'] != 'skipgram':
    #     raise NotImplementedError()
    # segmenter = WordSegmenter(args['segmenter'],
    #                           args['language'],
    #                           args['segmenter_len'])
    # sgm = segmenter.segment
    #
    # segm_voc_size = segmenter.unique_segments
    # word_segments = segmenter.max_len
    #
    # print("Max Word Len is %d segments" % word_segments)
    #
    # terminals = assemble_graph(model=args['model_name'],
    #                            vocab_size=args['vocabulary_size'],
    #                            segment_vocab_size=segm_voc_size,
    #                            max_word_segments=word_segments,
    #                            emb_size=args['dimensionality'])
    # else:
    if args['model_name'] == "fasttext":
        return Fasttext(vocab_size=args['vocabulary_size'],
                        emb_size=args['dimensionality'],
                        graph_path=args['graph_path'],
                        ckpt_path=args['ckpt_path'],
                        gpu_options=gpu_options,
                        segmenter_path=args['segmenter'],
                        max_segments=args['segmenter_len'])

    if args['model_name'] == "morph":
        return Morph(vocab_size=args['vocabulary_size'],
                     emb_size=args['dimensionality'],
                     graph_path=args['graph_path'],
                     ckpt_path=args['ckpt_path'],
                     gpu_options=gpu_options,
                     segmenter_path=args['segmenter'],
                     max_segments=args['segmenter_len'])

    if args['model_name'] == "morphgram":
        return MorphGram(vocab_size=args['vocabulary_size'],
                         emb_size=args['dimensionality'],
                         graph_path=args['graph_path'],
                         ckpt_path=args['ckpt_path'],
                         gpu_options=gpu_options,
                         segmenter_path=args['segmenter'],
                         max_segments=args['segmenter_len'])

    if args['model_name'] == "subwordcnn":
        return SubwordCNN(vocab_size=args['vocabulary_size'],
                          emb_size=args['dimensionality'],
                          graph_path=args['graph_path'],
                          ckpt_path=args['ckpt_path'],
                          gpu_options=gpu_options,
                          segmenter_path=args['segmenter'],
                          max_segments=args['segmenter_len'],
                          negative=args['negative'],
                          n_context=args['window_size'] * 2,
                          batch_size=args['batch_size']
                          )

    if args['model_name'] == "skipgram":
        return Skipgram(vocab_size=args['vocabulary_size'],
                        emb_size=args['dimensionality'],
                        graph_path=args['graph_path'],
                        ckpt_path=args['ckpt_path'],
                        gpu_options=gpu_options)

    # return terminals, sgm

# def save_snapshot(sess, saver, terminals, args):
#     batch_count = sess.run(terminals['batch_count'])
#     path = "./%s/%s_%d_%d" % (args['graph_path'],
#                               args['model_name'],
#                               args['vocabulary_size'],
#                               batch_count)
#     ckpt_p = "%s/model.ckpt" % path
#     assign_embeddings(sess, terminals, args)
#     _ = saver.save(sess, ckpt_p)

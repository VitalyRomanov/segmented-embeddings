from models import Fasttext

class Morph(Fasttext):
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 segmenter_path=None,
                 max_segments=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None,
                 model_name='morph'):

        super(Morph, self).__init__(vocab_size=vocab_size,
                 emb_size=emb_size,
                 segmenter_path=segmenter_path,
                 max_segments=max_segments,
                 graph_path=graph_path,
                 gpu_options=gpu_options,
                 ckpt_path=ckpt_path,
                 model_name=model_name)
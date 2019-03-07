#!/bin/bash

python stream_reader.py \
    -d 50 \
    -e 1 \
    -c 3 \
    -n 1 \
    -w 2 \
    -b 200 \
    -v 100 \
    -s 1e-4 \
    -m fasttext \
    -l en \
    -sgm n_gram_segmentation/en/ \
    -sgmlen 8 \
    -wiki 1 \
    -r 0 \
     "/Volumes/External/data_sink/en_wiki_tiny/" \
     "vocabularies/en/en_voc_tokenized.pkl"
#     | python stream_trainer.py

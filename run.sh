#!/bin/bash

python stream_reader.py \
    -d 50 \
    -e 1 \
    -c 200 \
    -n 10 \
    -w 2 \
    -b 2000 \
    -v 5000 \
    -s 1e-4 \
    -m fasttext \
    -l en \
    -sgm n_gram_segmentation/en/ \
    -sgmlen 15 \
    -wiki 1 \
    -r 0 \
     "/Volumes/External/data_sink/en_wiki_tiny/" \
     "vocabularies/en/en_voc_tokenized.pkl" \
     | python stream_trainer.py
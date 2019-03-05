#!/bin/bash

python stream_reader.py \
    -d 15 \
    -e 1 \
    -c 200 \
    -n 1 \
    -w 2 \
    -b 200 \
    -v 100 \
    -s 1e-4 \
    -m attentive \
    -l en \
    -sgm morpheme_segmentation/en/ \
    -wiki 1 \
    -r 0 \
     "/Volumes/Seagate 2nd part/en_wiki_tiny/" \
     "vocabularies/en/en_voc_tokenized.pkl" | python stream_trainer.py

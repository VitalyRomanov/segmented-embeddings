#!/bin/bash

python stream_reader.py \
    -d 150 \
    -e 1 \
    -c 20 \
    -n 20 \
    -w 2 \
    -b 2000 \
    -v 10000 \
    -s 1e-4 \
    -m skipgram \
    -l en \
    -sgmlen 8 \
    -wiki 1 \
    -r 0 \
     "/Volumes/External/datasets/Language/Corpus/en/en_wiki_tiny/" \
     | python stream_trainer.py

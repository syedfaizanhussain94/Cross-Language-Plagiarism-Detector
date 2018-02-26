#!/bin/bash
./MSR_corpus_compute_vector.sh
python dynamicPooling.py ALL
python train_classifier_on_MSR_corpus.py ALL
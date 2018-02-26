#!/bin/bash
DIR=`pwd -P`
export PYTHONPATH=$DIR/str2vec/src

python $PYTHONPATH/nn/rae_compute.py\
  instant/instantTestFile\
  generated_models/all_words_with_vectors\
  generated_models/RAE-model.model.gz\
  instant/instantVecFile.txt
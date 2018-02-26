#!/bin/bash
DIR=`pwd -P`
LANGPAIR='ALL' # Correct this
export PYTHONPATH=$DIR/str2vec/src

python $PYTHONPATH/nn/rae_compute.py\
  MSR_Corpus_parsed/MSRparaphrase_sentences_${LANGPAIR}_train\
  generated_models/all_words_with_vectors\
  generated_models/RAE-model.model.gz\
  generated_models/MSRparaphrase_sentences_${LANGPAIR}_train.vec.txt

python $PYTHONPATH/nn/rae_compute.py\
  MSR_Corpus_parsed/MSRparaphrase_sentences_${LANGPAIR}_test\
  generated_models/all_words_with_vectors\
  generated_models/RAE-model.model.gz\
  generated_models/MSRparaphrase_sentences_${LANGPAIR}_test.vec.txt
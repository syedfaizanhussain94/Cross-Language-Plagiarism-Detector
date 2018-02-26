#!/bin/bash
python generateAllWordsWithVectors.py 48
echo "STEP 1 COMPLETE: Generated all words with vectors"
python generateAllSentencesWithFrequencies.py
echo "STEP 2 COMPLETE: Generated all sentences with frequences"
./trainRAE.sh 1
echo "STEP 3 COMPLETE: RAE trained"
./MSR_corpus_compute_vector.sh
echo "STEP 4 COMPLETE: Vectors for Train and Test Senteces computed"
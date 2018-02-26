#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'uutsav'

import getopt
import sys
import os
import codecs
import string
import goslate	# google translate
import urllib2
import logging
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

sys.stdout = codecs.open('ble_out','w',encoding='utf-8-sig')

context_size = 48
min_word_count = 10
no_of_training_documents = "all"


# proxy_handler = urllib2.ProxyHandler({"http" : "http://ironport1.iitk.ac.in:3128"})
# proxy_opener = urllib2.build_opener(urllib2.HTTPHandler(proxy_handler), urllib2.HTTPSHandler(proxy_handler))

DEFAULT_STORY_INPUT_FILE = 'englishInputStory.txt'
DEFAULT_UNIQUE_WORDS_OUTPUT_FILE = 'englishUniqueWords.txt'
HINDI_UNIQUE_WORDS_FILE = "hindiUniqueWords.txt"

HINDI_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_en_TranslateGoldTruth_nouns.txt"
ENGLISH_TO_HINDI_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_hi_TranslateGoldTruth_nouns.txt"
GERMAN_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_en_TranslateGoldTruth_nouns.txt"
ENGLISH_TO_GERMAN_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_de_TranslateGoldTruth_nouns.txt"

HINDI_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_UniqueGoldTruth_nouns.txt"
ENGLISH_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_UniqueGoldTruth_nouns.txt"
GERMAN_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_UniqueGoldTruth_nouns.txt"


# HINDI_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_en_TranslateGoldTruth_verbs.txt"
# ENGLISH_TO_HINDI_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_hi_TranslateGoldTruth_verbs.txt"
# GERMAN_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_en_TranslateGoldTruth_verbs.txt"
# ENGLISH_TO_GERMAN_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_de_TranslateGoldTruth_verbs.txt"

# HINDI_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_UniqueGoldTruth_verbs.txt"
# ENGLISH_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_UniqueGoldTruth_verbs.txt"
# GERMAN_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_UniqueGoldTruth_verbs.txt"


# HINDI_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_en_TranslateGoldTruth_adj.txt"
# ENGLISH_TO_HINDI_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_hi_TranslateGoldTruth_adj.txt"
# GERMAN_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_en_TranslateGoldTruth_adj.txt"
# ENGLISH_TO_GERMAN_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_de_TranslateGoldTruth_adj.txt"

# HINDI_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_UniqueGoldTruth_adj.txt"
# ENGLISH_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_UniqueGoldTruth_adj.txt"
# GERMAN_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_UniqueGoldTruth_adj.txt"


# HINDI_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_en_TranslateGoldTruth_others.txt"
# ENGLISH_TO_HINDI_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_hi_TranslateGoldTruth_others.txt"
# GERMAN_TO_ENGLISH_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_en_TranslateGoldTruth_others.txt"
# ENGLISH_TO_GERMAN_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_de_TranslateGoldTruth_others.txt"

# HINDI_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/hi_UniqueGoldTruth_others.txt"
# ENGLISH_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/en_UniqueGoldTruth_others.txt"
# GERMAN_UNIQUE_TRANSLATE_NOT_WORKING = "BLE_test_cases/de_UniqueGoldTruth_others.txt"

# these are the files containg list of unique words in each language which would be used
# for testing purposes when translate api is not working

# These can be generated with:
# awk '{print $1}' en_hi_TranslateGoldTruth.txt > en_UniqueGoldTruth.txt
# awk '{print $1}' de_en_TranslateGoldTruth.txt > de_UniqueGoldTruth.txt
# awk '{print $1}' hi_en_TranslateGoldTruth.txt > hi_UniqueGoldTruth.txt


uniqueWords = set()


def genUniqueWordsFile(storyInputFileName, uniqueWordsOutputFileName):
    punctuations = list(string.punctuation)
    with codecs.open(storyInputFileName, 'r', encoding='utf-8-sig') as f:
        for line in f:
            # print "before: ", line
            for punkt in punctuations:
                if punkt in line:
                    line = line.replace(punkt, ' ')
            # print "after: ", line

            words = line.split()
            for word in words:
                uniqueWords.add(word)
        print ' # '.join(uniqueWords)

    g = codecs.open(uniqueWordsOutputFileName, 'w', encoding='utf-8-sig')
    g.write('\n'.join(uniqueWords))
    g.close()


def performBLETask(LG1_testWordFile, LG2_testWordFile, LG1Marker, LG2Marker):

	# Now load test words from LG1 say w1 and translate it to LG2 say w2
	# See the rank of this translated word in the nearest neighbours from wordvector 
	# model of w1

	# try:
		# gs = goslate.Goslate(opener=proxy_opener)
		# gs = goslate.Goslate()
		# gs.translate('Hello', 'hi')	# to test if google API is working
		# translateWorking = True
	# except:
	# since google translate is not working, load the previous translation from file
	# we can no longer use the new test files LG1_testWordFile and LG2_testWordFile
	print "Goslate API not working ..."
	print "Loading default test cases ..."
	translateWorking = False
	if LG1Marker == 'en':
		if LG2Marker == 'hi':
			LG1_goldTranslationFile = ENGLISH_TO_HINDI_TRANSLATE_NOT_WORKING
		else:
			LG1_goldTranslationFile = ENGLISH_TO_GERMAN_TRANSLATE_NOT_WORKING
		LG1_testWordFile = ENGLISH_UNIQUE_TRANSLATE_NOT_WORKING
	elif LG1Marker == 'hi':
		LG1_goldTranslationFile = HINDI_TO_ENGLISH_TRANSLATE_NOT_WORKING
		LG1_testWordFile = HINDI_UNIQUE_TRANSLATE_NOT_WORKING
	else:
		LG1_goldTranslationFile = GERMAN_TO_ENGLISH_TRANSLATE_NOT_WORKING
		LG1_testWordFile = GERMAN_UNIQUE_TRANSLATE_NOT_WORKING

	if LG2Marker == 'en':
		if LG1Marker == 'hi':
			LG2_goldTranslationFile = ENGLISH_TO_HINDI_TRANSLATE_NOT_WORKING
		else:
			LG2_goldTranslationFile = ENGLISH_TO_GERMAN_TRANSLATE_NOT_WORKING
		LG2_testWordFile = ENGLISH_UNIQUE_TRANSLATE_NOT_WORKING
	elif LG2Marker == 'hi':
		LG2_goldTranslationFile = HINDI_TO_ENGLISH_TRANSLATE_NOT_WORKING
		LG2_testWordFile = HINDI_UNIQUE_TRANSLATE_NOT_WORKING
	else:
		LG2_goldTranslationFile = GERMAN_TO_ENGLISH_TRANSLATE_NOT_WORKING
		LG2_testWordFile = GERMAN_UNIQUE_TRANSLATE_NOT_WORKING

	LG1_to_LG2_map = {}
	LG2_to_LG1_map = {}
	print LG1_testWordFile, " and ", LG2_testWordFile, " selected as the defualt 2 test files ..."
	with codecs.open(LG1_goldTranslationFile, 'r', encoding='utf-8-sig') as f:
		lines = f.readlines()
		for line in lines:
			line = line.rstrip('\n').lower()	# Check for hindi
			lst = line.split('\t')
			LG1_to_LG2_map[lst[0]] = lst[1:]

	with codecs.open(LG2_goldTranslationFile, 'r', encoding='utf-8-sig') as f:
		lines = f.readlines()
		for line in lines:
			line = line.rstrip('\n').lower()	# Check for hindi
			lst = line.split('\t')
			LG2_to_LG1_map[lst[0]] = lst[1:]

	# print LG1_to_LG2_map
	# print LG2_to_LG1_map

	no_neighbours = 25
	# load word2vec model
	# model_name='English-Hindi-vector-model-'+str(no_of_training_documents)+'-cs-'+str(context_size)+'-wc-'+str(min_word_count)
	model_name='English-Hindi-vector-model_with_words-'+str(no_of_training_documents)+'-cs-'+str(context_size)+'-wc-'+str(min_word_count)  
	model = word2vec.Word2Vec.load('models/' + model_name)


	score = (no_neighbours)*[0]
	c = 0
	with codecs.open(LG1_testWordFile, 'r', encoding='utf-8-sig') as f:
		for line in f:
			
			word = line.rstrip('\n').lower()
			# print word

			# if translateWorking:
			# 	translatedWords = gs.translate(word, LG2Marker)
			# else:
			# 	translatedWords = LG1_to_LG2_map[word]

			translatedWords = LG1_to_LG2_map[word]

			# Generate top 10 nearest neighbours of 'word'
			NN=[]
			if word in model.vocab:
				bestRank = 10000
				c += 1
				NN = [m[0] for m in model.most_similar(word,topn=no_neighbours)]
				for translatedWord in translatedWords:
					if translatedWord in NN:
						rank = NN.index(translatedWord)
						bestRank = min(bestRank, rank)

				if bestRank != 10000:
					score[bestRank] += 1

				if bestRank == 10000:
					bestRank = -1

				print "#", c, word, " : ", " ".join(translatedWords), " rank: ", bestRank, " neighbours: ", " ".join(NN)
			else:
				print word, " not in word2vec model"

	print "Rank stats: ", "for ", LG1Marker, " to ", LG2Marker
	for i in range(len(score)):
		print "#", i, "matches - ", score[i]

	accuracy = sum(score)
	print "Accuracy: ", accuracy, "/", c, " = ", float(accuracy)/float(c)
	print "\n\n\n"

	# reverse the process for the language pairs with roles interchanged
	score = no_neighbours*[0]
	c = 0
	with codecs.open(LG2_testWordFile, 'r', encoding='utf-8-sig') as f:
		for line in f:
			word = line.rstrip().lower()
			# print word

			# if translateWorking:
			# 	translatedWords = gs.translate(word, LG1Marker)
			# else:
			# 	translatedWords = LG2_to_LG1_map[word]

			translatedWords = LG2_to_LG1_map[word]

			# Generate top 10 nearest neighbours of 'word'
			if word in model.vocab:
				bestRank = 10000
				c += 1
				NN = [m[0] for m in model.most_similar(word,topn=no_neighbours)]
				for translatedWord in translatedWords:
					if translatedWord in NN:
						rank = NN.index(translatedWord)
						bestRank = min(bestRank, rank)

				if bestRank != 10000:
					score[bestRank] += 1

				if bestRank == 10000:
					bestRank = -1
					
				print "#", c, word, " : ", " ".join(translatedWords), " rank: ", bestRank, " neighbours: ", " ".join(NN)
			else:
				print word, " not in word2vec model"

	print "Rank stats: ", "for ", LG2Marker, " to ", LG1Marker
	for i in range(len(score)):
		print "#", i, "matches - ", score[i]
				
	accuracy = sum(score)
	print "Accuracy: ", accuracy, "/", c, " = ", float(accuracy)/float(c)

	return



def main(argv):
    storyInputFileName = None
    try:
        args = getopt.getopt(argv, '')
    except getopt.GetoptError:
        print 'usage : python BLE_task.py [filename]'
        sys.exit(2)

    for i in range(len(args[1])):
        if i == 0:
            storyInputFileName = args[1][i]
        elif i >= 1:
            print "Only 0 or 1 argument expected"
            print 'usage : python BLE_task.py [filename]'
            sys.exit(2)

    if storyInputFileName is None:
        storyInputFileName = DEFAULT_STORY_INPUT_FILE

    # genUniqueWordsFile(storyInputFileName, DEFAULT_UNIQUE_WORDS_OUTPUT_FILE)

    performBLETask(HINDI_UNIQUE_WORDS_FILE, DEFAULT_UNIQUE_WORDS_OUTPUT_FILE, 'hi', 'en')

main(sys.argv[1:])






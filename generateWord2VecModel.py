#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

import random
import re
import regex as fraudRe
import codecs
import nltk.data
import logging
import os
import subprocess
from subprocess import PIPE,Popen
import warnings
import sys
import argparse
import fnmatch
import cPickle
from BeautifulSoup import BeautifulStoneSoup
# from bs4 import BeautifulSoup
from gensim.models import word2vec
from random import shuffle

from nltk.stem.snowball import SnowballStemmer      # better than potter stemmer
ENstemmer = SnowballStemmer("english")
# ENstemmer_ignoreStopWords = SnowballStemmer("english", ignore_stopwords=True)


warnings.filterwarnings("ignore")

# sys.stdout = codecs.open('output','w',encoding='utf-8-sig')   # Write outout to a file named 'output'

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

parser = argparse.ArgumentParser(description='Train a Word2Vec model based on comparable corpora')
parser.add_argument('-c','--context_size', help='Integer for context window size',default=48)
args = vars(parser.parse_args())


# GLOBAL VARIABLES
epsilon = 1.0
context_size = args['context_size']
min_word_count = 10
no_of_training_documents = "all"
dir_name="generated_models"
# model_name = 'English-Hindi-vector-model-'+str(no_of_training_documents)+'-cs-'+str(context_size)+'-wc-'+str(min_word_count)   #PREVIOUS
model_name = 'English-Hindi-vector-model_with_words-'+str(no_of_training_documents)+'-cs-'+str(context_size)+'-wc-'+str(min_word_count)   #NEW
if no_of_training_documents == 'all':
    no_of_training_documents = -1

IF_STEMMING = False


def remove_punctuation(text):
    return fraudRe.sub(ur"\p{P}+", " ", text)

# tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
#                       if unicodedata.category(unichr(i)).startswith('P'))
# def remove_punctuation(text):
#     return text.translate(tbl)


def stemHindi(sentence):
    f = codecs.open('shallow_parser_input_file','w',encoding='utf-8-sig')
    f.write(sentence)
    f.close()

    bashCommand = "shallow_parser_hin shallow_parser_input_file shallow_parser_output_file"

    s = Popen(["shallow_parser_hin" ,"shallow_parser_input_file", "shallow_parser_output_file"], stderr=PIPE)

    _,err = s.communicate() # err  will be empty string if the program runs ok
    if err:
        print "As usual, shallow parser NOT WORKING !"
        print err
        return []

    # p1 = subprocess.call(bashCommand)
    # p1.wait()
    # os.system(bashCommand)
    ssfText = codecs.open('shallow_parser_output_file', encoding='utf-8-sig').read()

    words = parseStemmedHindi(ssfText)
 
    return words


def parseStemmedHindi(ssfText):
    # print ssfText
    # ssfWords = re.findall('^(((.*?)^))', ssfText, re.DOTALL|re.MULTILINE)
    ssflines = ssfText.split('\n')
    words = []
    for line in ssflines:
        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!", line
        num = re.search(r'\d+\.\d+', line)   # finding numbers of the form digits.digits
        if num:
            parts = line.split("af='")
            if len(parts) > 1:  # it should always happen !
                stem = parts[1].split(',')[0]
                # print stem
                words.append(stem)
    return words


def sentenceToWordlistHindi(sentence, remove_stopwords=False):
    soup = BeautifulStoneSoup(sentence)
    if soup is not None:
        sentence_text = soup.getText()
    else:
        print "soup has not been cooked yet !"
        return []

    words = sentence_text.split()

    # if IF_STEMMING:                           # don't do stemming here, shallow parser is worthless !
    #     words = stemHindi(sentence_text)


    if remove_stopwords:
        stops = set(stopwords.words("hindi"))
        words = [w for w in words if not w in stops]

    return words


def sentenceToWordlistEnglish(sentence, remove_stopwords=False):
    soup = BeautifulStoneSoup(sentence)
    if soup is not None:
        sentence_text = soup.getText()
    else:
        print "soup has not been cooked yet !"
        return []

    # 2. Remove non-letters
    # sentence_text = re.sub("[^a-zA-Z]"," ", sentence_text)

    words = sentence_text.lower().split()

    if IF_STEMMING:
        for i in range(len(words)):
            words[i] = ENstemmer.stem(words[i])


    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)


def documentToSentencesEnglish(document, remove_stopwords=False):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(document.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences.append(sentenceToWordlistEnglish( raw_sentence,remove_stopwords ))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def documentToSentencesHindi(document, remove_stopwords=False):
    # text=document.replace(u'।'.encode('utf-8'.decode('unicode-escape')),'.')      # if you have imported codecs, 
    # text=text.replace(u'॥'.encode('utf-8'.decode('unicode-escape')),'.')          # dont do these
    text=document.replace(u'।','.')
    text=text.replace(u'॥','.')

    raw_sentences=re.split(r'[?,.,!]',text,flags=re.UNICODE)
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(sentenceToWordlistHindi(raw_sentence, remove_stopwords))
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences    


def fileToSentencesEnglish(file):
    document = codecs.open(file, encoding='utf-8-sig').read()
    sentences = documentToSentencesEnglish(document)
    return sentences


def fileToSentencesHindi(file):
    document = codecs.open(file, encoding='utf-8-sig').read()
    sentences = documentToSentencesHindi(document)
    return sentences


def fileToWordsEnglish(fileName):
    document = codecs.open(fileName, encoding='utf-8-sig').read()
    # print "Before punctuation: "
    # print document
    document = remove_punctuation(document)
    # document = re.sub(r'[^\w\s]','',document)     # does not work for unicode string
    # print "After punctuation: "
    # print document
    words = sentenceToWordlistEnglish(document)
    # print "After stemming: "
    # print " ".join(words)
    # Shortcomings of stemming: industry -> industri, retired -> retiri, united kingdom -> unit kingdom
    return words


def fileToWordsHindi(fileName):
    if IF_STEMMING:
        ssfText = codecs.open("hindi/comparable_hindi_stemmed/"+fileName.split('/')[-1], encoding='utf-8-sig').read()
        words = parseStemmedHindi(ssfText)
    else:
        document = codecs.open(fileName, encoding='utf-8-sig').read()
        # remove punctuation
        # document = document.replace(u'।', '') # no need, the remove_punctuation function is good enough
        # document = document.replace(u'॥', '')
        # print "Before punctuation: "
        # print document
        document = remove_punctuation(document)
        # print "After punctuation: "
        # print document
        words = sentenceToWordlistHindi(document)

    return words


def docRandomShuffleWithWordLists(doc1, doc2):
    # doc1 = [item for sublist in document1 for item in sublist] 
    # doc2 = [item for sublist in document2 for item in sublist]

    no_words_doc1 = len(doc1)
    no_words_doc2 = len(doc2)
    if no_words_doc2 == 0 or no_words_doc1 == 0:
        print "WARNING: no words in file"
        return []

    words_ratio = float(no_words_doc1)/float(no_words_doc2)
    if words_ratio < 1:
        words_ratio = 1/words_ratio

    if not (abs(words_ratio-1.0) < epsilon):
        print "skewed file word lengths", no_words_doc1, " and ", no_words_doc2
        # print "adding in ratio", no_large, "and", no_small
        return []

    if no_words_doc1 > no_words_doc2:
        no_large = no_words_doc1
        # no_large = int(no_words_doc2*(1.0+epsilon))
        no_small= no_words_doc2
        large_doc = doc1
        # large_doc = doc1[:no_large]
        small_doc = doc2 
    else:
        no_large = no_words_doc2
        # no_large = int(no_words_doc1*(1.0+epsilon))
        no_small = no_words_doc1
        large_doc = doc2
        # large_doc = doc2[:no_large]
        small_doc = doc1

    mergedDocument = []
    combined_doc = large_doc+small_doc
    
    shuffle(combined_doc)

    # The below code is time consuming way of randomly shuffling python list
    # no_words_doc1 = len(document1)
    # no_words_doc2 = len(document2)
    # total_words = no_words_doc1+no_words_doc2
    # while total_words > 0:
    #     r = random.randint(0,total_words-1)
    #     random_word = combined_doc[r]
    #     mergedDocument.append(random_word)
    #     del combined_doc[r]             # remove by index
    #     total_words = total_words-1

    return combined_doc


def lengthRatioShuffle(document1, document2):
    no_words_doc1 = len(document1)
    no_words_doc2 = len(document2)

    if no_words_doc2 == 0 or no_words_doc1 == 0:
        print "WARNING: no words in file"
        return []

    words_ratio = float(no_words_doc1)/float(no_words_doc2)
    if words_ratio < 1:
        words_ratio = 1/words_ratio

    if not (abs(words_ratio-1.0) < epsilon):
        print "skewed file word lengths", no_words_doc1, " and ", no_words_doc2
        return []

    total_words = no_words_doc1+no_words_doc2
    if(no_words_doc1 > no_words_doc2):
        large_doc = document1
        small_doc = document2
        # ratio = no_words_doc1/no_words_doc2
    else:
        large_doc = document2
        small_doc = document1
        # ratio = no_words_doc2/no_words_doc1
    # r = int(round(ratio))
    k = 0
    mergedDocument = []
    for i in range(len(small_doc)):
        # print i
        mergedDocument.append(small_doc[i])
        start = int(k+1)
        end = int(k+words_ratio+1)
        mergedDocument.extend(large_doc[start:end])
        k = k+words_ratio

    start = int(k+1)
    mergedDocument.extend(large_doc[start:len(large_doc)])

    # print " ".join(small_doc)
    # print " ".join(large_doc)
    # print " ".join(mergedDocument)

    return mergedDocument


def orderPreservedRandomShuffle(document1, document2):
    no_words_doc1 = sum(1 for sentence in document1 for word in sentence)
    no_words_doc2 = sum(1 for sentence in document2 for word in sentence)
    words_ratio = float(no_words_doc1)/float(no_words_doc2)
    # print words_ratio, abs(words_ratio-1.0),epsilon
    # print "NO OF WORDS IN DOC1 and DOC2: ",no_words_doc1,no_words_doc2
    if not (abs(words_ratio-1.0) < epsilon):
        print "Not adding"
        return []

    if no_words_doc1 > no_words_doc2:
        no_large = no_words_doc1
        no_small = no_words_doc2
        large_doc = document1
        small_doc_words = [item for sublist in document2 for item in sublist] 
    else:
        no_large = no_words_doc2
        no_small = no_words_doc1
        large_doc = document2
        small_doc_words = [item for sublist in document1 for item in sublist] 

    j_final = 0
    count_words = 0
    j = 0
    large_doc_copy = large_doc

    random_indices = sorted([random.randint(0,no_large) for i in range(no_small)])
    # print random_indices
    for i in range(len(random_indices)):
        if(count_words <= random_indices[i]):  # sentence change
            for j in range(j_final,len(large_doc)):
                count_words = count_words+len(large_doc[j])
                if(count_words >= random_indices[i]):
                    break
        large_doc[j].insert(random_indices[i]-(count_words-len(large_doc_copy[j])),small_doc_words[i])
        j_final = j+1
    return large_doc


def orderPreservedRandomShuffleWithWordLists(document1, document2):
    no_words_doc1 = len(document1)
    no_words_doc2 = len(document2)
    if no_words_doc2 == 0 or no_words_doc1 == 0:
        print "WARNING: no words in file"
        return []

    words_ratio = float(no_words_doc1)/float(no_words_doc2)
    if words_ratio < 1:
        words_ratio = 1/words_ratio

    

    if no_words_doc1 > no_words_doc2:
        # no_large = no_words_doc1
        no_large = int(no_words_doc2*(1.0+epsilon))
        no_small= no_words_doc2
        # large_doc = document1
        large_doc = document1[:no_large]
        small_doc = document2 
    else:
        # no_large = no_words_doc2
        no_large = int(no_words_doc1*(1.0+epsilon))
        no_small = no_words_doc1
        # large_doc = document2
        large_doc = document2[:no_large]
        small_doc = document1


    # print words_ratio, abs(words_ratio-1.0),epsilon
    # print "NO OF WORDS IN DOC1 and DOC2: ",no_words_doc1,no_words_doc2
    if not (abs(words_ratio-1.0) < epsilon):
        print "skewed file word lengths", no_words_doc1, " and ", no_words_doc2
        # print "adding in ratio", no_large, "and", no_small
        return []
        # print no_large, " ".join(large_doc)
        # print no_small, " ".join(small_doc)


    random_indices = sorted([random.randint(0,no_large) for i in range(no_small)])
    # print "random:",random_indices
    for i in range(len(random_indices)):
         large_doc.insert((random_indices[i]+i),small_doc[i])

    return large_doc


def trainWord2Vec(sentences,model_name):

    # Set values for various parameters
    num_features = 200    # Word vector dimensionality
    min_count = min_word_count   # Minimum word count
    num_workers = 2      # Number of threads to run in parallel
    context = context_size          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    sg = 1 # Use skip-gram instead of CBOW (default is 1)
    negative = 25 # How many negative words to sample

    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_count, \
                window = context, sample = downsampling, sg = sg, \
                negative = negative)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(dir_name+'/'+model_name)
    # model = Word2Vec.load('vec_model')
    
    return model


# Trains a word2vec model for the English German pair, from K documents each and writes the 
# the set of words in both the languages into files 
def trainEnglishGermanWordVec(K, model_name, lang1_set_file_name, lang2_set_file_name, lang1_corpus_dir, lang2_corpus_dir):
    sents_lang1 = []
    sents_lang2 = []
    sentences = []
    words_lang1 = []
    words_lang2 = []
    # corpusDirName = "English"
    corpusDirName = lang1_corpus_dir
    k = 0
    taken_documents = 0
    # lang1_words_file = codecs.open(set_file_name1, 'w',encoding='utf-8-sig')
    # lang2_words_file = codecs.open(set_file_name2, 'w',encoding='utf-8-sig')
    lang1_words_file = open(dir_name+'/'+lang1_set_file_name, 'wb',)
    lang2_words_file = open(dir_name+'/'+lang2_set_file_name, 'wb',)
    for root, dirnames, lang1FileNames in os.walk(corpusDirName):
        lang1FileNames = sorted(lang1FileNames)
        # print lang1FileNames
        for lang1FileName in lang1FileNames:
            
            if k == K:
                break
            k += 1
            id = id =''.join(x for x in lang1FileName if x.isdigit())
            lang2FileName = 'hi_'+id
            file1 = lang1_corpus_dir+'/'+lang1FileName
            file2 = lang2_corpus_dir+'/'+lang2FileName

            print k

            # IN THE NEW IMPLEMENTATION sents_lang1 = wordsLang1..
            # NEW: Using fileToWords instead of fileToSentences

            #PREVIOUS
            # sents_lang1=fileToSentencesEnglish(file1)
            # # sents_lang2=fileToSentencesEnglish(file2)
            # sents_lang2=fileToSentencesHindi(file2)

            # wordsLang1 = [item for sublist in sents_lang1 for item in sublist]
            # wordsLang2 = [item for sublist in sents_lang2 for item in sublist]
            # sentence=orderPreservedRandomShuffle(sents_lang1,sents_lang2)


            if IF_STEMMING:
                # since we are doing stemming which is time consuming (thanks to shallow parser)
                # so we better get a rough estimate of words in both languages before we do stemming
                # as we are anyways discarding files outside epsilon boundary
                size1 = os.path.getsize(file1) 
                size2 = os.path.getsize(file2) 
                # print size1, "###", size2
                size_ratio = float(size1)/float(size2)
                if size_ratio < 1:
                    size_ratio = 1/size_ratio
                if not (abs(size_ratio-1.0) < epsilon):
                    print file1, " and ", file2, " are of skewed size ratio ", size1, " and ", size2
                    continue


            #NEW
            sents_lang1 = fileToWordsEnglish(file1)
            # sents_lang2 = fileToWordsEnglish(file2)
            sents_lang2 = fileToWordsHindi(file2)
            # print "sent_lang2",sents_lang2

            wordsLang1 = sents_lang1
            wordsLang2 = sents_lang2

            # print " ".join(wordsLang1)
            # print "$$$$$$$$$$$$$$$$$$$$$$$\n\n"
            # print " ".join(wordsLang2)
           
            # sentence = orderPreservedRandomShuffleWithWordLists(sents_lang1, sents_lang2)
            # sentence = docRandomShuffleWithWordLists(sents_lang1, sents_lang2)
            sentence = lengthRatioShuffle(sents_lang1, sents_lang2)
          
            if not len(sentence) == 0:
                print "extending lengths ", len(sents_lang1), "and", len(sents_lang2)
                words_lang1.extend(wordsLang1)
                words_lang2.extend(wordsLang2)
                # sentences.extend(sentence)            # PREVIOUS: Using the previous method: splitting on sentence level
                sentences.append(sentence)              # NEW: Splitting on document level
                
                taken_documents += 1
                # print " ".join(wordsLang1)
                # print "\n\n"
                # print " ".join(wordsLang2)
                # wordsMerged=[item for sublist in sentence for item in sublist]
                # print "\n\n"
                # print " ".join(wordsMerged)
                # print " ".join(sentence) 

    print "Total Documents = ", taken_documents

    lang1_words_set = set(words_lang1)
    lang2_words_set = set(words_lang2)

    cPickle.dump(lang1_words_set, lang1_words_file)
    cPickle.dump(lang2_words_set, lang2_words_file)
    lang1_words_file.close()
    lang2_words_file.close()

    model = trainWord2Vec(sentences, model_name)
    # model = ""
    return model


def main():

    lang1_words_file_name = "english_words"
    lang2_words_file_name = "hindi_words"

    lang1_corpus_dir = "comparable_english-hindi_wiki_data/comparable_english_articles"
    lang2_corpus_dir = "comparable_english-hindi_wiki_data/comparable_hindi_articles"
    model = trainEnglishGermanWordVec(no_of_training_documents, model_name, lang1_words_file_name, lang2_words_file_name, lang1_corpus_dir, lang2_corpus_dir)
    # model = word2vec.Word2Vec.load('models'/+model_name)
    # # print '\n\n\n'

    words_in_language_1_file = open(lang1_words_file_name, 'rb')
    words_in_language_2_file = open(lang2_words_file_name, 'rb')
    words_lang2 = cPickle.load(words_in_language_2_file)
    words_lang1 = cPickle.load(words_in_language_1_file)

    # print " ".join(m[0] for m in model.most_similar("queen",topn=50))
    # print model.most_similar("wasser",topn=50)
    # print model.most_similar("the",topn=50)
    # print model.most_similar("die",topn=50)
    # print model.most_similar("germany",topn=50)
    # print model.most_similar("deutschland",topn=50)
    # print model.doesnt_match("mann woman Frau kitchen".split())
    # for word in words:
    #     print word
    #     if word[0] in words_lang2:
    #         print "GERMAN"
    #     if word[0] in words_lang1:
    #         print "ENGLISH"


main()

def oldMain():
    # print random.sample(range(100),10)
    # document1=["I","want","an","icecream","from","the","shop"]
    # document2=["This","is","cool","and","hey","yes"]
    # print orderPreservedRandomShuffleWithWordLists(document1,document2)
    # # doc= lengthRatioShuffle(document1,document2)
    doc="I saw a small, cat! Oh! shit...#%;''"
    doc = re.sub(r'[^\w\s]','',doc)
    print doc
    # print doc

# oldMain()




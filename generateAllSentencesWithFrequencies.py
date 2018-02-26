#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

import random,re,codecs,nltk,logging,os,warnings, sys,fnmatch, cPickle
from bs4 import BeautifulSoup
from gensim.models import word2vec
from collections import Counter

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def documentToSentencesEnglish( document, remove_stopwords=False ):
    document = document.replace('\n', ' ').replace('\r', '')
    raw_sentences = tokenizer.tokenize(document.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(raw_sentence)
    return sentences

def documentToSentencesHindi(document, remove_stopwords=False):
    document = document.replace('\n', ' ').replace('\r', '')
    text=document.replace(u'।','.')
    text=text.replace(u'॥','.')
    raw_sentences=re.split(r'[?,.,!]',text,flags=re.UNICODE)
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(raw_sentence)
    return sentences    


def fileToSentencesEnglish(file):
    document=codecs.open(file,encoding='utf-8-sig').read()
    sentences=documentToSentencesEnglish(document)
    return sentences

def fileToSentencesHindi(file):
    document=codecs.open(file,encoding='utf-8-sig').read()
    sentences=documentToSentencesHindi(document)
    return sentences

lang1_corpus_dir="comparable_english-hindi_wiki_data/comparable_english_articles"
lang2_corpus_dir="comparable_english-hindi_wiki_data/comparable_hindi_articles"
corpusDirName=lang1_corpus_dir
combined_sentences=[]
iter_i=0
iter_j=0
for root, dirnames, lang1FileNames in os.walk(corpusDirName):
	# print "OUTER ITERATION: ",iter_i
	iter_i+=1
        # print lang1FileName
	for lang1FileName in lang1FileNames:
		print "INNER ITERATION: ",iter_j
		iter_j+=1
	    	id= id=''.join(x for x in lang1FileName if x.isdigit())
		lang2FileName='hi_'+id
		file1=lang1_corpus_dir+'/'+lang1FileName
		file2=lang2_corpus_dir+'/'+lang2FileName
		sents_lang1=fileToSentencesEnglish(file1)
		sents_lang2=fileToSentencesHindi(file2)
		for i in range(len(sents_lang1)):
			sss=sents_lang1[i]
			ssss=sss.lstrip()
			# print ssss
			s=ssss.lower()
			ss = re.sub(r'[^\w\s]','',s)
			# print s
			sents_lang1[i]=ss

		for i in range(len(sents_lang2)):
			sss=sents_lang2[i]
			ssss=sss.lstrip()
			# print ssss
			s=ssss.lower()
			ss = re.sub(r',.','',ssss)
			# print ss
			sents_lang2[i]=ss
		# print len(sents_lang1), len(sents_lang2)
		combined_sentences.extend(sents_lang1)
		combined_sentences.extend(sents_lang2)
	   	if iter_j == 250:
        		break
   	break
        # print len(combined_sentences)
# print combined_sentences
freqs = Counter(combined_sentences)
dir_name="generated_models"
f=codecs.open(dir_name+"/all_sentences_with_frequencies","w",encoding='utf-8-sig')
# itr=0
for word_freq in freqs.items():
   	# itr+=1
	s=word_freq[0]
	string_to_write=s+' ||| '+str(word_freq[1]) +'\n'
	f.write(string_to_write)
# print "TOTAL NUMBER OF SENTENCES = ",itr
f.close()



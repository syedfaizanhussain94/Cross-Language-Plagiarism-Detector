#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
import codecs

def generateMSRCrossLingualFiles():

	# GENERATES THE 16 FILES BELOW
	MSRparaphrase_sentences_train_file_HH=codecs.open("MSRparaphrase_sentences_HH_train",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_HH=codecs.open("MSRparaphrase_sentences_HH_test",encoding="utf-8-sig")
	MSRparaphrase_sentences_train_file_EE=codecs.open("MSRparaphrase_sentences_EE_train",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_EE=codecs.open("MSRparaphrase_sentences_EE_test",encoding="utf-8-sig")

	reverse_MSRparaphrase_sentences_train_file_HH=codecs.open("reverse_MSRparaphrase_sentences_HH_train","w",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_HH=codecs.open("reverse_MSRparaphrase_sentences_HH_test","w",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_train_file_EE=codecs.open("reverse_MSRparaphrase_sentences_EE_train","w",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_EE=codecs.open("reverse_MSRparaphrase_sentences_EE_test","w",encoding="utf-8-sig")

	MSRparaphrase_sentences_train_file_EH=codecs.open("MSRparaphrase_sentences_EH_train","w",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_EH=codecs.open("MSRparaphrase_sentences_EH_test","w",encoding="utf-8-sig")
	MSRparaphrase_sentences_train_file_HE=codecs.open("MSRparaphrase_sentences_HE_train","w",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_HE=codecs.open("MSRparaphrase_sentences_HE_test","w",encoding="utf-8-sig")

	reverse_MSRparaphrase_sentences_train_file_EH=codecs.open("reverse_MSRparaphrase_sentences_EH_train","w",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_EH=codecs.open("reverse_MSRparaphrase_sentences_EH_test","w",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_train_file_HE=codecs.open("reverse_MSRparaphrase_sentences_HE_train","w",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_HE=codecs.open("reverse_MSRparaphrase_sentences_HE_test","w",encoding="utf-8-sig")
		
	# READS ONLY THE TWO BELOW FILES
	MSRparaphrase_sentences_train_HH=MSRparaphrase_sentences_train_file_HH.readlines()
	MSRparaphrase_sentences_train_EE=MSRparaphrase_sentences_train_file_EE.readlines()	

	for i in range(0,len(MSRparaphrase_sentences_train_EE),2):
		sentE1=MSRparaphrase_sentences_train_EE[i]
		sentE2=MSRparaphrase_sentences_train_EE[i+1]
		sentH1=MSRparaphrase_sentences_train_HH[i]
		sentH2=MSRparaphrase_sentences_train_HH[i+1]

		reverse_MSRparaphrase_sentences_train_file_EE.write(sentE2+sentE1)
		reverse_MSRparaphrase_sentences_train_file_HH.write(sentH2+sentH1)

		MSRparaphrase_sentences_train_file_EH.write(sentE1+sentH2)
		MSRparaphrase_sentences_train_file_HE.write(sentH1+sentE2)
		reverse_MSRparaphrase_sentences_train_file_EH.write(sentE2+sentH1)
		reverse_MSRparaphrase_sentences_train_file_HE.write(sentH2+sentE1)

	MSRparaphrase_sentences_test_EE=MSRparaphrase_sentences_test_file_EE.readlines()
	MSRparaphrase_sentences_test_HH=MSRparaphrase_sentences_test_file_HH.readlines()

	for i in range(0,len(MSRparaphrase_sentences_test_EE),2):
		sentE1=MSRparaphrase_sentences_test_EE[i]
		sentE2=MSRparaphrase_sentences_test_EE[i+1]
		sentH1=MSRparaphrase_sentences_test_HH[i]
		sentH2=MSRparaphrase_sentences_test_HH[i+1]

		reverse_MSRparaphrase_sentences_test_file_EE.write(sentE2+sentE1)
		reverse_MSRparaphrase_sentences_test_file_HH.write(sentH2+sentH1)

		MSRparaphrase_sentences_test_file_EH.write(sentE1+sentH2)
		MSRparaphrase_sentences_test_file_HE.write(sentH1+sentE2)
		reverse_MSRparaphrase_sentences_test_file_EH.write(sentE2+sentH1)
		reverse_MSRparaphrase_sentences_test_file_HE.write(sentH2+sentE1)


	MSRparaphrase_sentences_train_file_ALL=codecs.open("MSRparaphrase_sentences_ALL_train","w",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_ALL=codecs.open("MSRparaphrase_sentences_ALL_test","w",encoding="utf-8-sig")

def mergeToAllFile():
	MSRparaphrase_sentences_train_file_ALL=codecs.open("MSRparaphrase_sentences_ALL_train","w",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_ALL=codecs.open("MSRparaphrase_sentences_ALL_test","w",encoding="utf-8-sig")

	MSRparaphrase_sentences_train_file_HH=codecs.open("MSRparaphrase_sentences_HH_train",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_HH=codecs.open("MSRparaphrase_sentences_HH_test",encoding="utf-8-sig")
	MSRparaphrase_sentences_train_file_EE=codecs.open("MSRparaphrase_sentences_EE_train",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_EE=codecs.open("MSRparaphrase_sentences_EE_test",encoding="utf-8-sig")

	reverse_MSRparaphrase_sentences_train_file_HH=codecs.open("reverse_MSRparaphrase_sentences_HH_train",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_HH=codecs.open("reverse_MSRparaphrase_sentences_HH_test",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_train_file_EE=codecs.open("reverse_MSRparaphrase_sentences_EE_train",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_EE=codecs.open("reverse_MSRparaphrase_sentences_EE_test",encoding="utf-8-sig")

	MSRparaphrase_sentences_train_file_EH=codecs.open("MSRparaphrase_sentences_EH_train",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_EH=codecs.open("MSRparaphrase_sentences_EH_test",encoding="utf-8-sig")
	MSRparaphrase_sentences_train_file_HE=codecs.open("MSRparaphrase_sentences_HE_train",encoding="utf-8-sig")
	MSRparaphrase_sentences_test_file_HE=codecs.open("MSRparaphrase_sentences_HE_test",encoding="utf-8-sig")

	reverse_MSRparaphrase_sentences_train_file_EH=codecs.open("reverse_MSRparaphrase_sentences_EH_train",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_EH=codecs.open("reverse_MSRparaphrase_sentences_EH_test",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_train_file_HE=codecs.open("reverse_MSRparaphrase_sentences_HE_train",encoding="utf-8-sig")
	reverse_MSRparaphrase_sentences_test_file_HE=codecs.open("reverse_MSRparaphrase_sentences_HE_test",encoding="utf-8-sig")

	train1=MSRparaphrase_sentences_train_file_EE.read()
	train2=MSRparaphrase_sentences_train_file_HH.read()
	train3=reverse_MSRparaphrase_sentences_train_file_HH.read()
	train4=reverse_MSRparaphrase_sentences_train_file_EE.read()
	train5=MSRparaphrase_sentences_train_file_EH.read()
	train6=MSRparaphrase_sentences_train_file_HE.read()
	train7=reverse_MSRparaphrase_sentences_train_file_EH.read()
	train8=reverse_MSRparaphrase_sentences_train_file_HE.read()

	train_ALL=train1+train2+train3+train4+train5+train6+train7+train8

	test1=MSRparaphrase_sentences_test_file_EE.read()
	test2=MSRparaphrase_sentences_test_file_HH.read()
	test3=reverse_MSRparaphrase_sentences_test_file_HH.read()
	test4=reverse_MSRparaphrase_sentences_test_file_EE.read()
	test5=MSRparaphrase_sentences_test_file_EH.read()
	test6=MSRparaphrase_sentences_test_file_HE.read()
	test7=reverse_MSRparaphrase_sentences_test_file_EH.read()
	test8=reverse_MSRparaphrase_sentences_test_file_HE.read()
	# print len(test6)

	test_ALL=test1+test2+test3+test4+test5+test6+test7+test8


	MSRparaphrase_sentences_train_file_ALL.write(train_ALL)
	MSRparaphrase_sentences_test_file_ALL.write(test_ALL)

generateMSRCrossLingualFiles()
mergeToAllFile()
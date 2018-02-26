#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'uutsav'

import getopt
import sys
import glob
import os
import fnmatch
import re
from BeautifulSoup import BeautifulStoneSoup
import codecs
import shutil


DEFAULT_CORPUS_DIR = "english_extracted"
CORRESPONDING_ENGLISH_SAVE_DIR = "comparable_english_articles"
CORRESPONDING_HINDI_SAVE_DIR = "comparable_hindi_articles"
HINDI_ARTICLES_DIR = "hindi_articles"
invalid_filename_chars = [';', ':', '/', '\\', '*', '?', '[', ']', ')', '(', '{', '}', '.', '|', '=', ',']
hindiIdEnglishTitleFile = "hindiIdEnglishTitle"     # can be given by the command line user also

comparableFileNumber = 0    # The comparable english hindi articles will be stored as file name  en_/hi_comparableFileNumber
comparableMap = dict()      # it is a map from english title -> corresponding hindi id


# populates the comparableMap with the mapping of english title -> corresponding hindi id
def genComparableMap():
    global comparableMap
    with codecs.open(hindiIdEnglishTitleFile, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # ignore the first line containing headers
            line = line.rstrip('\n')
            lst = line.split('\t')
            # now we need to ensure that we have the hindi article with id = lst[0] in our HINDI_ARTICLES_DIR
            # otherwise we won't add it to our map
            if os.path.isfile(HINDI_ARTICLES_DIR+'/'+lst[0]):
                comparableMap[lst[1]] = lst[0]
            # print "map of ##"+lst[1]+"##", "is", comparableMap[lst[1]]
    return


def scanWikiExtractorFile(fileName):
    global comparableFileNumber
    global comparableMap
    print "reading file ", fileName
    with open(fileName, 'r') as f:
        content = f.read()
        articles = content.split('</doc>')
        for article in articles[:-1]:
            # print article
            try:
                soup = BeautifulStoneSoup(article)
            except UnicodeEncodeError:
                print "UnicodeEncodeError"
                pass
                continue

            title = soup.find('doc')['title']
            if title in comparableMap:  # this english article has a corresponding hindi article

                print title, 'found'

                comparableFileNumber += 1
                text = soup.getText()

                # Why to go into pain of renaming title since we are not storing file by title anyways !!
                # Now title can contain '/', ':' and all such characters which make file names invalid
                # for c in invalid_filename_chars:
                #     if c in title:
                #         title = title.replace(c, ' ')
                # print title
                # engArtName = title

                g = codecs.open(CORRESPONDING_ENGLISH_SAVE_DIR+'/'+'en_' + str(comparableFileNumber), 'w', encoding='utf-8-sig')
                g.write(text)
                g.close()

                # now we need to move the comparable hindi article into the CORRESPONDING_HINDI_SAVE_DIR

                # But why didn't we move all hindi articles existing in the comparableMap into CORRESPONDING_HINDI_SAVE_DIR
                # anyways instead of doing it one by one now.
                # This is because all those hindi ids may not have their corresponding english articles in this wiki dump
                # Moreover, since we have to document align english with hindi, so we would first have had to store
                # the comparableFileNumber beforehand for all hindi english articles and use it here instead of generating
                # a new one. It could have led to gaps in b/w if some english articles could not be found !

                hindiId = comparableMap[title]
                shutil.copyfile(HINDI_ARTICLES_DIR+'/'+hindiId, CORRESPONDING_HINDI_SAVE_DIR+'/'+'hi_'+str(comparableFileNumber))

                # Now remove this mapping from dictionary since all keys are unique, so we won't get this again.
                # But removing keys would definitely reduce size of map and fasten up the worst case lookup time
                del comparableMap[title]

    return


def genCorrespondingFiles(corpusDirName):

    matches = []
    for root, dirnames, filenames in os.walk(corpusDirName):
        for filename in fnmatch.filter(filenames, 'wiki_*'):
            matches.append(os.path.join(root, filename))

    matches = sorted(matches)
    # i = 0
    for fileName in matches:
        # if i > 1:
            # break
        scanWikiExtractorFile(fileName)
        # i += 1
    return


# This script takes a English corpus directory which have separate xml wikipedia dumps
# It also takes a HindiIdEnglishTitle file, ie a file which maps Hindi wiki Id's to English wiki Titles
# Then we would exhaustively search the English corpus dictionary and write only those
# english articles in a separate file which are present in this hindi to english map

# These corresponding english articles would be saved in CORRESPONDING_ENGLISH_SAVE_DIR
# The comparable Hindi articles would also be copied to CORRESPONDING_HINDI_SAVE_DIR
def main(argv):
    corpusDirName = None
    try:
        args = getopt.getopt(argv, '')
    except getopt.GetoptError:
        print 'usage : python corresponding_english_document_generator.py [directory name] [hindi-ID-English-Title file]'
        sys.exit(2)

    for i in range(len(args[1])):
        if i == 0:
            corpusDirName = args[1][i]
        elif i == 1:
            hindiIdEnglishTitleFile = args[1][i]
        elif i >= 2:
            print "Only 1 or 2 argument expected"
            print 'usage: python corresponding_english_document_generator.py [directory name] [hindi-ID-English-Title file]'
            sys.exit(2)

    if corpusDirName is None:
        corpusDirName = DEFAULT_CORPUS_DIR

    genComparableMap()
    genCorrespondingFiles(corpusDirName)


main(sys.argv[1:])









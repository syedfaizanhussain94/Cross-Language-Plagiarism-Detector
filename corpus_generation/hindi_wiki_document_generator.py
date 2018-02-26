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


DEFAULT_CORPUS_DIR = "hindi_extracted"
HINDI_SAVE_DIR = "hindi_articles"
invalid_filename_chars = [';', ':', '/', '\\', '*', '?', '[', ']', ')', '(', '{', '}', '.', '|', '=', ',']
HINDI_ID_TITLE_MAP_FILE = "hindi_id_to_title"


# Takes a wiki xml file and breaks it up into several articles and save them into separate files
# The files would be named according to the title of the article
def scanWikiExtractorFile(fileName):
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
            text = soup.getText()
            # Now title can contain '/', ':' and all such characters which make file names invalid
            for c in invalid_filename_chars:
                if c in title:
                    title = title.replace(c, ' ')

            print title
            hindiArtName = title
            g = codecs.open(HINDI_SAVE_DIR+'/'+hindiArtName, 'w', encoding='utf-8-sig')
            g.write(text)
            g.close()
            # break
    return

#  This takes a corpus directory as input, recursively go to each sub directory
#  and finds all xml wiki files. It then passes these xml to the article splitter
#  function scanWikiExtractorFile() one by one to actually generate all articles
#  as separate files
def genSeparateFiles(corpusDirName):

    matches = []
    for root, dirnames, filenames in os.walk(corpusDirName):
        for filename in fnmatch.filter(filenames, 'wiki_*'):
            matches.append(os.path.join(root, filename))

    matches = sorted(matches)

    for fileName in matches:
        scanWikiExtractorFile(fileName)
    return


# generates a map of Id and its corresponding wiki title and store it in a file
def genIdTitleMap(corpusDirName):

    matches = []
    for root, dirnames, filenames in os.walk(corpusDirName):
        for filename in fnmatch.filter(filenames, 'wiki_*'):
            matches.append(os.path.join(root, filename))
    matches = sorted(matches)

    g = codecs.open(HINDI_ID_TITLE_MAP_FILE, 'w', encoding='utf-8-sig')

    for fileName in matches:
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
                id = soup.find('doc')['id']
                # Now title can contain '/', ':' and all such characters which make file names invalid
                for c in invalid_filename_chars:
                    if c in title:
                        title = title.replace(c, ' ')

                print id, title
                row = id + '\t' + title + '\n'
                g.write(row)

    g.close()
    return


def renameHindiTitlesToIds():
    with codecs.open(HINDI_ID_TITLE_MAP_FILE, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\n')
            lst = line.split('\t')
            print lst[0], lst[1]
            try:
                print lst[1], 'renamed to ', lst[0]
                os.rename(HINDI_SAVE_DIR+'/'+lst[1], HINDI_SAVE_DIR+'/'+'uutsav_'+lst[0])   # to avoid renaming to files already in the HINDI_SAVE_DIR
            except:
                print lst[0], "does not exist"
                pass
            # break

    # now we need to remove the uutsav_ prefix from each file name
    for root, dirnames, filenames in os.walk(HINDI_SAVE_DIR):
        for filename in fnmatch.filter(filenames, 'uutsav_*'):
            modified_filename = filename.replace('uutsav_', '')
            print modified_filename
            os.rename(HINDI_SAVE_DIR+'/'+filename, HINDI_SAVE_DIR+'/'+modified_filename)
            # break
    return


# This script takes a Hindi corpus directory which have separate xml wikipedia dumps
# It separates out all the articles of each of these xmls into separate text files
# and store it in HINDI_SAVE_DIR

# It also contains a function genIdTitleMap() which generates a map of Hindi Id and
# its corresponding Hindi wiki title and store it in a file HINDI_ID_TITLE_MAP_FILE

# This file would be used for cross searching when we would have hindi Id's and English
# title from the wiki sql dump. Then we would need to know which of these hindi articles
# generated by genSeparateFiles() could actually be used in comparable corpus

def main(argv):
    corpusDirName = None
    try:
        args = getopt.getopt(argv, '')
    except getopt.GetoptError:
        print 'usage: python hindi_wiki_document_generator.py [directory name]'
        sys.exit(2)

    for i in range(len(args[1])):
        if i == 0:
            corpusDirName = args[1][i]
        elif i >= 1:
            print "Only 0 or 1 argument expected"
            print 'usage: python hindi_wiki_document_generator.py [directory name]'
            sys.exit(2)

    if corpusDirName is None:
        corpusDirName = DEFAULT_CORPUS_DIR

    # genSeparateFiles(corpusDirName)

    # genIdTitleMap(corpusDirName)

    renameHindiTitlesToIds()

main(sys.argv[1:])




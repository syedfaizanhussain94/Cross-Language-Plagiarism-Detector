#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-


import os,re
from BeautifulSoup import BeautifulStoneSoup



# Parse the single XML file of English-German document aligned wikpedia corpus
# to generate a single file English-Hindi-id which has html content inside
def wikiExtractComparableCorpus(fileName):  
    flag = 0
    with open(fileName) as f:
        for line in f:
            if not (flag == 1):
                # print line
                soup = BeautifulSoup(line)
                found = soup.find('articlepair')
                if(found):
                    id = found['id']
                    print id
                    # exit()
                    g = open("English-German/EnglishGerman"+str(id),"w")
                    g.write(line)
                    flag = 1 
            elif(flag == 1):
                g.write(line)
                if(re.search(r'</articlePair>', line)): 
                    flag = 0
                    g.close()


# Takes the English-German-id file and splits it into two separate English and German
# files after BeautifulSoup has extracted the content from it
def splitSingleFileIntoEnglishGerman(fileName):
    id = ''.join(x for x in fileName if x.isdigit())
    f = open(fileName).read()
    articles = f.split('</article>')
    flag = 0
    for article in articles[:-1]:
        try:
            soup = BeautifulSoup(article)
        except UnicodeEncodeError:
            print "SOUP ERROR"
            pass
            continue
        text = soup.getText()
        if flag == 0:
            with codecs.open('German/German'+id,"w",encoding="utf-8-sig") as g:
                g.write(text)
        else:
             with codecs.open('English/English'+id,"w",encoding="utf-8-sig") as g:
                g.write(text)
        flag = not flag

# Solits all the combined English-German-id files into English and German single files
def splitEnglishGermanFiles():
    matches = []
    k = 0
    corpusDirName='English-German'
    for root, dirnames, filenames in os.walk(corpusDirName):
        print "READING ",k
        for filename in fnmatch.filter(filenames, 'EnglishGerman*'):
            matches.append(os.path.join(root, filename))
        k += 1

    # matches = sorted(matches)
    k = 0
    for fileName in matches:
        print "WRITING ",k
        # print fileName
        # if fileName == 'hindi_extracted/AA/wiki_15':
        #     print 'inside'
        splitSingleFileIntoEnglishGerman(fileName)
        k += 1

def main():
    f="wikicomp-2014_deen.xml"
    wikiExtractComparableCorpus(f)
    splitEnglishGermanFiles()

main()

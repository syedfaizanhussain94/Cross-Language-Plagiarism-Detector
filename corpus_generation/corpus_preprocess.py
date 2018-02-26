__author__ = 'uutsav'

import random
import re
from gensim.models import word2vec


def fileToWords(fileName,lang):
    fil=open(fileName).readlines()
    stuff=[]
    for line in fil:
        line=line.strip('\n').strip('\r')
        # line=re.sub(r'[0-9]*',"",line)
        # line=re.sub(r'[.$*&-?\!$%_+\=-`"\'<>~@\(\)\'"]*',"",line)
        # for un in unwanted:
        #     line=line.replace(un.encode('utf-8'.decode('unicode-escape')) ,"")
        if lang== "english":
            if re.search(r'[a-z]', line):
                stuff.extend(line.strip(" ").split(" "))
        elif lang == "hindi" or lang=="sanskrit":
            if not re.search(r'[a-z]', line):
                line=line.split(".")

                for lin in line:
                    lin=lin.lstrip().strip(" ").split(" ")
                    for li in lin:
                        if li==" " or li=="  " or li=="" :
                            continue
                        stuff.append(li)
    return stuff


def naiveShuffle(document1, document2):
    mergedDocument=[]
    no_words_doc1=len(document1)
    no_words_doc2=len(document2)
    combined_doc=document1+document2
    total_words=no_words_doc1+no_words_doc2
    while(total_words>0):
        r= random.randint(0,total_words-1)
        random_word=combined_doc[r]
        mergedDocument.append(random_word)
        del combined_doc[r]             # remove by index
        total_words = total_words-1
    return mergedDocument


def orderPreservedShuffle(document1, document2):
    no_words_doc1=len(document1)
    no_words_doc2=len(document2)
    if no_words_doc1 > no_words_doc2:
        no_large=no_words_doc1
        no_small=no_words_doc2
        large_doc=document1
        small_doc=document2
    else:
        no_large=no_words_doc2
        no_small=no_words_doc1
        large_doc=document2
        small_doc=document1
    random_indices= sorted(random.sample(range(no_large), no_small))
    # print random_indices
    for i in range(len(random_indices)):
        large_doc.insert(random_indices[i],small_doc[i])
    return large_doc


def lengthRatioShuffle(document1, document2):
    no_words_doc1=len(document1)
    no_words_doc2=len(document2)
    total_words=no_words_doc1+no_words_doc2
    if(no_words_doc1 > no_words_doc2):
        large_doc=document1
        small_doc=document2
        ratio=no_words_doc1/no_words_doc2
    else:
        large_doc=document2
        small_doc=document1
        ratio=no_words_doc2/no_words_doc1
    r=int(round(ratio))
    k=0
    mergedDocument=[]
    for i in range(len(small_doc)):
        print i
        mergedDocument.append(small_doc[i])
        mergedDocument.extend(large_doc[k:k+r])
        k=k+r
    mergedDocument.extend(large_doc[k:len(large_doc)])
    return mergedDocument


def trainWord2Vec(sentences,model_name):
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(model_name)
    # model = Word2Vec.load('vec_model')

    return model


def oldMain():
    # print random.sample(range(100),10)
    document1=["I","want","an","icecream","from","the","shop"]
    document2=["This","is","cool"]
    doc=[document1,document2]
    # doc= lengthRatioShuffle(document1,document2)


    print doc


    file='text8'
    sent=[]
    with open(file) as f:
        for word in f:
            print word
            break
            sent.append(word)
    # print sent[0][1]
    print len(sent)
    mdoc=[]
    for i in range(len(doc)):
       # final.append = [w for w in doc[i]]
        s=''
        for word in doc[i]:
            s=s+' '+word
        mdoc.append(s)
    print mdoc
    # print sent
    # exit()
    model_name="vec_model"
    model=trainWord2Vec(mdoc,model_name)

# file="bgita.txt"
# hindDoc= fileToWords(file,"hindi")
# print len(hindDoc)
# engDoc=["I","want","an","icecream","from","the","shop"]
# print orderPreservedShuffle(engDoc,hindDoc)

e=[["i want tt"],["sds dsd sds sdd"],["sds dsd sds sdd"],["sds dsd sds sdd"],["sds dsd sds sdd"],["sds dsd sds sdd"],["sds dsd sds sdd"],["sds dsd sds sdd"]]
trainWord2Vec(e,"ds")




























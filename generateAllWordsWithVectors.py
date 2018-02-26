import logging,cPickle,codecs,sys
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


context_size =  int(sys.argv[1])
# corpus =sys.argv[2]
# alphabet =sys.argv[3]

# context_size=35
min_word_count=10
no_of_training_documents='all'
# no_of_training_documents=200

dir_name="generated_models"
model_name='English-Hindi-vector-model_with_words-'+str(no_of_training_documents)+'-cs-'+str(context_size)+'-wc-'+str(min_word_count)
model = word2vec.Word2Vec.load(dir_name+'/'+model_name)

lang1_words_file_name="english_words"
lang2_words_file_name="hindi_words"
words_in_language_1_file = open(dir_name+'/'+lang1_words_file_name, 'rb')
words_in_language_2_file = open(dir_name+'/'+lang2_words_file_name, 'rb')
words_lang2=list(cPickle.load(words_in_language_2_file))
words_lang1=list(cPickle.load(words_in_language_1_file))


f=codecs.open(dir_name+"/all_words_with_vectors","w",encoding='utf-8-sig')
for word in words_lang1:
	string_to_write=word
	try:
		mod=model[word]
	except KeyError,e :
		mod=model['unknown']
		pass
		continue
	for x in mod:
		string_to_write = string_to_write+" "+str(x) 
	string_to_write=string_to_write+'\n'
	f.write(string_to_write)

for word in words_lang2:
	string_to_write=word
	try:
		mod=model[word]
	except KeyError,e :
		mod=model['unknown']
		pass
		continue
	for x in mod:
		string_to_write = string_to_write+" "+str(x) 
	string_to_write=string_to_write+'\n'
	f.write(string_to_write)

f.close()

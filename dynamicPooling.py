import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cPickle,sys

dir_name="generated_models"

def minPool(A):
	return np.matrix(A).min()

def dynamicPooling(A,m,n,Np):
	final_matrix=np.zeros((Np,Np))
	subset_row=np.zeros(Np)
	subset_col=np.zeros(Np)
	min_row_subset=np.floor_divide(m,Np)
	min_col_subset=np.floor_divide(n,Np)

	rem_row_subset=np.mod(m,Np)
	rem_col_subset=np.mod(n,Np)

	subset_row.fill(min_row_subset)
	subset_col.fill(min_col_subset)

	# print subset_row, subset_col
	for i in range(Np-1,-1,-1):
		if rem_row_subset > 0:
			subset_row[i]+=1
			rem_row_subset =rem_row_subset-1
		if rem_col_subset > 0:
			subset_col[i]+=1
			rem_col_subset =rem_col_subset-1
		if rem_col_subset == 0 and rem_row_subset ==0:
			break


	subset_row=[int(s) for s in subset_row]
	subset_col=[int(s) for s in subset_col]		
	# print "Subset row:", subset_row
	# print "subset col:",subset_col
	ini_row=0
	ini_col=0
	for i in range(Np):
		ini_col=0
		for j in range(Np):
			# print "iter",i,j
			final_matrix[i][j]=minPool(A[ini_row:ini_row+subset_row[i],ini_col:ini_col+subset_col[j]])
			ini_col=ini_col+subset_col[j]
		ini_row=ini_row+subset_row[i]
	return final_matrix

def dumpFinalVectors(sents_file_name,labels_file_name,X_file_name,Y_file_name,Np=7):
	# MSR_train_sents=open(path_sents_vecs+train_sents_file_name,"r").readlines()
	# MSR_train_labels=open(path_labels+train_labels_file_name,"r").readlines()

	print "OPENING FILES"
	# MSR_sents=open(dir_name+'/'+sents_file_name,"r").readlines()
	# MSR_labels=open(dir_name+'/'+labels_file_name,"r").readlines()
	# print "FILES OPENED"

	X=[]
	Y=[]
	with open(path_labels+'/'+labels_file_name,"r") as MSR_labels:
		for line in MSR_labels:
			label=int(line)
			Y.append(label)
	print "LABLES VARIABLE UPDATED"

	iteration_count=0
	with open(dir_name+'/'+sents_file_name,"r") as MSR_sents:
		for line in MSR_sents:
			iteration_count+=1
			print "ITERATION: ",iteration_count
			if iteration_count%2!=0:
				sentence1=line
			else:
				sentence2=line
				# sentence1=MSR_sents[i]
				# sentence2=MSR_sents[i+1]
				sent1_preprocess_list=sentence1.split(';')
				# print len(sentence1_vectors)
				sent2_preprocess_list=sentence2.split(';')
				# print len(sentence1_vectors)
				sent1_string_vectors=[s.split() for s in sent1_preprocess_list]
				sent1_vectors = [list(map(float, row)) for row in sent1_string_vectors]
				sent2_string_vectors=[s.split() for s in sent2_preprocess_list]
				sent2_vectors = [list(map(float, row)) for row in sent2_string_vectors]

				# print len(sent1_vectors), len(sent2_vectors)
				# sent1_vectors =np.array(sent1_vectors)
				# sent2_vectors =np.array(sent2_vectors)
				similarity_matrix=np.zeros((len(sent1_vectors),len(sent2_vectors)))

				for i in range(len(sent1_vectors)):
					for j in range(len(sent2_vectors)):
						similarity_matrix[i][j]=cosine_similarity(sent1_vectors[i],sent2_vectors[j])

				m=len(sent1_vectors)	#no_rows
				n=len(sent2_vectors)	#no_columns
				similarity_matrix=np.matrix(similarity_matrix)
				# print "INITITAL SM SHAPE: ",similarity_matrix.shape
				row_diff=Np-m
				# print "row diff:",row_diff
				col_diff=Np-n
				for i in range(row_diff):
					similarity_matrix=np.vstack((similarity_matrix,similarity_matrix[i,:]))
				print similarity_matrix.shape
				# print "col diff:",col_diff
				for i in range(col_diff):
					similarity_matrix=np.hstack((similarity_matrix,similarity_matrix[:,i]))
				# print "PADDED SM SHAPE:",similarity_matrix.shape
				m=similarity_matrix.shape[0]
				n=similarity_matrix.shape[1]
				print "m,n",m,n
				pooledMatrix=dynamicPooling(similarity_matrix,m,n,Np)
				# print "POOLED SM SHAPE:",pooledMatrix.shape

				X.append(pooledMatrix)

	X=np.array(X)
	print X.shape
	# print X
	# print Y
	X_file=open(dir_name+'/'+X_file_name,"w")
	Y_file=open(dir_name+'/'+Y_file_name,"w")
	cPickle.dump(X,X_file)
        cPickle.dump(Y,Y_file)

	# DUMPING INTO FILE

lang_pair =  sys.argv[1]
# path_sents_vecs='/home/enayat/Academics/SEM7/NLP/Project/str2vec-master/demo-data/str2vec-demo/output/'
path_labels='MSR_Corpus_parsed'
train_sents_file_name='MSRparaphrase_sentences_'+str(lang_pair)+'_train.vec.txt'
train_labels_file_name='MSRparaphrase_labels_'+str(lang_pair)+'_train'
test_sents_file_name='MSRparaphrase_sentences_'+str(lang_pair)+'_test.vec.txt'
test_labels_file_name='MSRparaphrase_labels_'+str(lang_pair)+'_test'

Np=15
X_train_file_name="X_train"
Y_train_file_name="Y_train"
X_test_file_name="X_test"
Y_test_file_name="Y_test"

dumpFinalVectors(train_sents_file_name,train_labels_file_name,X_train_file_name,Y_train_file_name, 15)
dumpFinalVectors(test_sents_file_name,test_labels_file_name,X_test_file_name,Y_test_file_name,15)
# A=np.zeros((11,15))
# print dynamicPooling(A,15,15,15)

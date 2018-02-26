from subprocess import call
import codecs,os
from svmutil import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cPickle
from sklearn import preprocessing

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
	
def generatePooledMatrix(sentence1,sentence2,Np):
	sent1_preprocess_list=sentence1.split(';')
	# print len(sentence1_vectors)
	sent2_preprocess_list=sentence2.split(';')
	# print len(sentence1_vectors)
	sent1_string_vectors=[s.split() for s in sent1_preprocess_list]
	sent1_vectors = [list(map(float, row)) for row in sent1_string_vectors]
	sent2_string_vectors=[s.split() for s in sent2_preprocess_list]
	sent2_vectors = [list(map(float, row)) for row in sent2_string_vectors]

	print len(sent1_vectors), len(sent2_vectors)
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
	X=[]
	X.append(pooledMatrix)
	return X
	# return pooledMatrix

def main():
	sent1=raw_input("ENTER SENTENCE 1: ")
	sent2=raw_input("ENTER SENTENCE 2: ")
	dir_name="generated_models"
	# sent1="hello how are you?"
	# sent2="hey how do you do?"

	Np=15
	# pwd=os.getcwd()

	f=codecs.open("instant/instantTestFile","w",encoding="utf-8-sig")
	f.write(sent1+'\n'+sent2+'\n')
	f.close()

	a=["./instant_compute_vector.sh"]
	call(a)

	vec_file=open("instant/instantVecFile.txt").readlines()
	print "len:",len(vec_file)
	sentence1=vec_file[0]
	sentence2=vec_file[1]
	
	X=generatePooledMatrix(sentence1,sentence2,Np)
	X=np.array(X)

	X =[preprocessing.scale(x) for x in X]	# ZERO MEAN AND UNIT VARIANCE
	X = [np.asarray(x).reshape(-1).tolist() for x in X]	# CONVNVERTING MATRIX TO ARRAY

	# print "len of X:",len(X)

	model_name="libsvm_model"
	m=svm_load_model(dir_name+'/'+model_name)
	print svm_predict([1],X,m)

main()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import cPickle,re
import numpy as np
import liblinearutil as ll
from svmutil import *
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn import decomposition,grid_search
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#besr params
c= 10
gamma= 0.1
dir_name="generated_models"

def simpleLibLinear(X_train,Y_train):
	prob = ll.problem(Y_train,X_train)
	param = ll.parameter('-c '+str(c))
	m = ll.train(prob, param)
	return m	

def simpleLibSVM(X_train,Y_train,model_name):
	prob = svm_problem(Y_train,X_train)
	# queit mode used : -q
	# param = svm_parameter('-t 0 -c 4 -v '+str(kCV))
	# param = svm_parameter('-t 2 -c 4 -v '+str(kCV)+' -g '+str(sigma)) # k cross-validation
	# m = svm_train(prob, param)	# k cross-validation
	# param = svm_parameter('-t 2 -c 4 -q -g '+str(sigma))
	# param = svm_parameter('-t 0 -c '+str(c)+' -q')		# LINEAR
	param = svm_parameter('-t 2 -c '+str(c)+' -q -g '+str(gamma))			# RBF
	# param = svm_parameter('-t 1 -d 2 -c '+str(c)+' -q -r '+str(coef0))  # QUADRATIC
	# param = svm_parameter('-t 0 -c 8 -q')
	m = svm_train(prob, param)
	svm_save_model(dir_name+'/'+model_name, m)
	return m

def f1(sent1,sent2):
	num1=re.findall("[-+]?\d+[\.]?\d*", sent1)
	num2=re.findall("[-+]?\d+[\.]?\d*", sent2)
	num1=[float(n) for n in num1]
	num2=[float(n) for n in num2]
	# print num1,num2
	if num1 ==num2:
		# print "works"
		return 1
	else:
		return 0

def f2(sent1, sent2):
	num1=re.findall("[-+]?\d+[\.]?\d*", sent1)
	num2=re.findall("[-+]?\d+[\.]?\d*", sent2)
	num1=[float(n) for n in num1]
	num2=[float(n) for n in num2]
	return int(any(i in num1 for i in num2))	#IF TWO SENTENCES HAVE AY COMMON ELEMENTS

def f3(sent1, sent2):
	num1=re.findall("[-+]?\d+[\.]?\d*", sent1)
	num2=re.findall("[-+]?\d+[\.]?\d*", sent2)
	num1=[float(n) for n in num1]
	num2=[float(n) for n in num2]
	return int((not Counter(num1) - Counter(num2)) or (not Counter(num2) - Counter(num1)))

def f4(sent1,sent2):
	return abs(len(sent1.split())-len(sent2.split()))

def f5(sent1,sent2):
	words1=sent1.split()
	words2=sent2.split()
	common=[element for element in words1 if element in words2]
	return float(len(common))/len(words1)

def f6(sent1,sent2):
	words1=sent1.split()
	words2=sent2.split()
	common=[element for element in words2 if element in words1]
	return float(len(common))/len(words2)

def main():
	lang_pair = sys.argv[1]

	X_train_file_name="X_train"
	Y_train_file_name="Y_train"
	X_test_file_name="X_test"
	Y_test_file_name="Y_test"

	X_train_file=open(dir_name+'/'+X_train_file_name,"r")
	Y_train_file=open(dir_name+'/'+Y_train_file_name,"r")
	X_test_file=open(dir_name+'/'+X_test_file_name,"r")
	Y_test_file=open(dir_name+'/'+Y_test_file_name,"r")

	x_train=cPickle.load(X_train_file)
	Y_train=cPickle.load(Y_train_file)
	x_test=cPickle.load(X_test_file)
	Y_test=cPickle.load(Y_test_file)

	#Doubling the zero labels
	len_Y_train=len(Y_train)
	x_train=x_train.tolist()
	# Y_train=Y_train.tolist()
	
	# SCALING TO ZERO MEAN AND UNIT VARIANCE:
	x_train = [preprocessing.scale(x) for x in x_train]
	x_test =[preprocessing.scale(x) for x in x_test]

	# Use arrays instead of matrix for X
	x_train = [np.asarray(x).reshape(-1).tolist() for x in x_train]
	x_test = [np.asarray(x).reshape(-1).tolist() for x in x_test]

	sentences_train=open('MSR_Corpus_parsed/MSRparaphrase_sentences_'+str(lang_pair)+'_train',"r").readlines()
	sentences_test=open('MSR_Corpus_parsed/MSRparaphrase_sentences_'+str(lang_pair)+'_test',"r").readlines()

	X_train=[]
	X_test=[]
	# X_train.append(x_train)
	# X_test.append(x_test)
	# do

	for j in range(0,len(sentences_train),2):
		# print i
		sent1=sentences_train[j]
		sent2=sentences_train[j+1]
		i=j/2
		
		F1=f1(sent1,sent2)
		F2=f2(sent1,sent2)
		F3=f3(sent1,sent2)
		F4=f4(sent1,sent2)
		F5=f5(sent1,sent2)
		F6=f6(sent1,sent2)
		# f7=f7(sent1,sent2)
	
		# x=[x_train[i],F1,F2,F3,F4,F5,F6]
		x=[F1,F2,F3,F4,F5,F6]
		# print x
		# X_train[i].append(F1)
		# X_train[i].append(F2)
		# X_train[i].append(F3)
		# X_train[i].append(F4)
		# X_traino[i].append(F5)
		# X_train.append(F6)
		# X_train.append(x)
		x_train[i].extend(x)

	for j in range(0,len(sentences_test),2):
		sent1=sentences_test[j]
		sent2=sentences_test[j+1]
		i=j/2
		
		F1=f1(sent1,sent2)
		F2=f2(sent1,sent2)
		F3=f3(sent1,sent2)
		F4=f4(sent1,sent2)
		F5=f5(sent1,sent2)
		F6=f6(sent1,sent2)
		# F7=f7(sent1,sent2)

		# x=[x_test[i],F1,F2,F3,F4,F5,F6]
		x=[F1,F2,F3,F4,F5,F6]
		# print x
		# X_test.append(F1)
		# X_test.append(F2)
		# X_test.append(F3)
		# X_test.append(F4)
		# X_test.append(F5)
		# X_test.append(F6)
		# X_test.append(x)
		x_test[i].extend(x)

	X_train=x_train
	X_test=x_test
	print len(X_train),len(Y_train),len(X_test),len(Y_test)
	k=0
	for i in range(len_Y_train):
		if Y_train[i]==0 and k%2 ==0:
			X_train.append(X_train[i])
			Y_train.append(Y_train[i])
		k+=1
	
	Y_count=Counter(Y_train)
	print "Y_train count:",Y_count
	# print np.array(X_train).shape
	# print X_train

	# X_train=X_train.tolist()
	# X_test=X_test.tolist()

	# ---------------------------------- LIBSVM --------------------------------------------	
	# print "USING LIBSVM"
	model=simpleLibSVM(X_train,Y_train,"libsvm_model")
	print "Accuracy using Libsvm on MSR test data:",svm_predict(Y_test,X_test,model)[1]
	# --------------------------------------------------------------------------------------

	# ---------------------------------TUNING A LIBSVM----------------------------------------

	# print "GRID SEARCH"
	# svc = SVC(kernel='rbf')
	# C=np.linspace(0.01, 100, num=10)
	# gamma=np.linspace(0.01, 100, num=5)
	# tuned_parameters = [
	#                 {'C': C,'gamma':gamma}]                
	# clf = grid_search.GridSearchCV(estimator=svc,param_grid=tuned_parameters)
	# clf.fit(X_train,Y_train)
	# Y_pred = clf.predict(X_test)
	# print Y_pred
	# print 'ACCURACY SCORE ON TEST: %0.3f' % accuracy_score(Y_test, Y_pred)
	# best_params= clf.best_params_
	# print "BEST PARAM FOR THE ITERATION: ",best_params

	# max_score=0.0
	# best_c=0.0


	# for params, mean_score, scores in clf.grid_scores_:
	# 	print("%0.3f (+/-%0.03f) for %r"
	# 	% (mean_score, scores.std() * 2, params))
	# 	if mean_score>max_score:
	# 		max_score=mean_score
	# 		best_c=best_params['C']

	# --------------------------------------------------------------------------------------

	# ----------------------------- LOGISTIC REGRESSION-------------------------------------

	# print "USING LOGISTIC REGRESSION"

	# TUNING FOR LOGISTIC REGRESSIION REGRESSION

	# C=np.linspace(0.01, 100, num=10)
	# tuned_parameters = [{'C': C}] 
	# model=LogisticRegression()               
	# clf = grid_search.GridSearchCV(estimator=model,param_grid=tuned_parameters)
	# clf.fit(X_train,Y_train)
	# best_params= clf.best_params_
	# print "BEST PARAM FOR THE ITERATION: ",best_params
	# max_score=0.0
	# best_c=0.0

	# for params, mean_score, scores in clf.grid_scores_:
	# 	print("%0.3f (+/-%0.03f) for %r"
	# 	% (mean_score, scores.std() * 2, params))
	# 	if mean_score>max_score:
	# 		max_score=mean_score
	# 		best_c=best_params['C']
	# print max_score, best_c


	# model=LogisticRegression(penalty='l1',C=0.01)
	# model.fit(X_train,Y_train)

	# Y_pred = model.predict(X_test)
	# Y_count=Counter(Y_pred)
	# print "Pred Y Count: ",Y_count
	# Y_count=Counter(Y_test)
	# print "Test Y Count: ",Y_count
	# # print Y_pred
	# print 'ACCURACY SCORE: %0.3f' % accuracy_score(Y_test, Y_pred)
	# print confusion_matrix(Y_test,Y_pred)

	# --------------------------------------------------------------------------------------

	# ---------------------------------- LIBLINEAR -----------------------------------------

	# print "USING LIBLINEAR"
	# model=simpleLibLinear(X_train,Y_train)
	# ll.predict(Y_test,X_test,model)

	# --------------------------------------------------------------------------------------

	

main()
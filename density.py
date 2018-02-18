# -*- coding:utf-8 -*- 
from __future__ import division
import numpy as np
import pandas as pd
import cPickle as Pickle
import os,sys
# from math import log
from numpy import log

from sklearn import metrics
from nltk.corpus import stopwords


from GlobalConvergence import Judge, intiRho_embedding, intiRho_onehot
from util import slide_window, load_bin_vec, briany_test_file
from evaluation import evaluation_plus, mrr_metric, map_metric
import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read('config.conf')
dirs, file_path, train_files, dev_files, test_files, gobal_var, _, _, _, _ = cf.sections()

train_fname = cf.get(file_path, 'train_fname')
train_density_sim_fname = cf.get(train_files, 'density_sim')

test_fname = cf.get(file_path, 'test_fname')
test_density_sim_fname = cf.get(test_files, 'density_sim')


# np.set_printoptions(threshold='nan')

word_count = [0]
random_word_count = [0]

# stopwordlist = stopwords.words('english')
stopwordlist = []

def getlM_q(sen,laplace=0, dependence=True ):
	probs={}
	for char in [i for i in sen.split() if i not in stopwordlist]:
		probs.setdefault(char,laplace)
		probs[char]+=1

	return probs

def getlM_a(sen, q_dict, laplace=0, dependence=True ):
	probs={}
	q_sen = [i for i in sen.split() if i in q_dict.keys()]
	for char in q_sen:
		probs.setdefault(char,laplace)
		probs[char]+=1

	# if len(probs) == 0:

	return probs

def getDependence(sen, q_dict, laplace=0):
	probs={}
	for word in slide_window(sen, q_dict, 9):
		key = "#".join(word)
		probs.setdefault(key,laplace)
		probs[key]+=1
	return probs


def getDensity( probs_term , probs_dependence, worddict={}):
	input_dict=dict()
	weights=sum(probs_term.values()) +sum (probs_dependence.values())
	dim = len(worddict)
	wordset = np.array(worddict.keys())
	for word,weight in probs_term.items():
		index= np.where(wordset == word)
		a= np.zeros(dim)
		a[index]=1;
		input_dict[word]= [weight/weights, np.outer(a, a)/np.inner(a, a)]
	for word,weight in probs_dependence.items():
		chars=word.split("#")
		lenght= len(chars)
		a=np.zeros(dim)
		for char in chars:
			index= np.where(wordset == char)
			a[index]+=(1.0/lenght)**0.5;
			# a += worddict[char]/lenght
		try:
			input_dict[word][0] += weight/weights
		except:
			input_dict[word]= [ weight/weights, np.outer(a, a)/np.inner(a, a)]
	rhoM0 =intiRho_onehot(probs_term, wordset)
	if len(input_dict) == 0:
		input_dict = {'####':[0, np.outer(np.zeros(dim), np.zeros(dim))]}
	rhoM = Judge(rhoM0, input_dict)
	return rhoM

def PairDensity(qapair, wf = {}):
	question = qapair['question']
	answer = qapair['answer']

	question_term = getlM_q(question)
	worddict = question_term
	probs_dependence=getDependence(question, worddict)

	answer_term = getlM_a(answer, worddict)
	a_probs_dependence = getDependence(answer, worddict)

	# a_probs_dependence = {}
	# probs_dependence = {}

	quesitonRho= getDensity(question_term, probs_dependence, worddict)
	answerRho=getDensity(answer_term, a_probs_dependence, worddict)

	# Pickle.dump([qapair['qid'], qapair['aid'], quesitonRho, answerRho, qapair['flag']], wf)

	q_len = len(question.split())
	a_len = len([a for a in answer.split() if a in question.split()])
	score = np.trace(np.dot(quesitonRho, answerRho))
	# score = np.trace(np.dot(100 * quesitonRho, answerRho * (a_len / (a_len * q_len + 1e-5)))) 
	return score

def build_voc_set(df):
	res = set()
	for index in range(len(df)):
		res = res | set(df.iloc[index]['question'].split())
		res = res | set(df.iloc[index]['answer'].split())
	return res


def get_density(rng):
	
	df_train= pd.read_csv(train_fname, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)
	df_test= pd.read_csv(test_fname, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)

	df_train['question'] = df_train['question'].str.lower()
	df_train['answer'] = df_train['answer'].str.lower()
	df_test['question'] = df_test['question'].str.lower()
	df_test['answer'] = df_test['answer'].str.lower()


	train_scores = []

	print 'train crous begin to training'
	for index in range(len(df_train)):
		if index % 100 == 0:
			print 'train have been completed ', index
		item = df_train.iloc[index]
		train_score = PairDensity(item)
		train_scores.append(train_score)

	df_train['score'] = train_scores

	train_density_sim = open(train_density_sim_fname, 'wb')
	Pickle.dump(train_scores, train_density_sim)
	train_density_sim.close()

	print metrics.roc_auc_score(df_train['flag'], train_scores) * 100
	print df_train.groupby('question').apply(mrr_metric).mean()
	print df_train.groupby('question').apply(map_metric).mean()

	test_scores = []

	print 'test crous begin to training'
	for index in range(len(df_test)):
		if index % 100 == 0:
			print 'test have been completed ', index
		item = df_test.iloc[index]
		test_score = PairDensity(item)
		test_scores.append(test_score)

	df_test['score'] = test_scores

	test_density_sim = open(test_density_sim_fname, 'wb')
	Pickle.dump(test_scores, test_density_sim)
	test_density_sim.close()

	# briany_test_file(df_test)
	# briany_test_file(df_train, 'train')

	
	print metrics.roc_auc_score(df_test['flag'], test_scores) * 100
	test_mrr = df_test.groupby('question').apply(mrr_metric).mean()
	test_map = df_test.groupby('question').apply(map_metric).mean()
	print test_mrr
	print test_map



	


if __name__ == '__main__':

	rng = np.random.RandomState(23455)
	get_density(rng)
	print word_count[0]
	print random_word_count[0]
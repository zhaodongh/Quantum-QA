from __future__ import division
import numpy as np 
import os,time
import cPickle as Pickle
from math import log
# from numba import autojit, jit

rng = np.random.RandomState(234)


dim = 50

def timeDeco(fun):
	def wrapper(*args ,**kwargs):
		start=time.time()
		result=fun( *args ,**kwargs)
		end=time.time()
		print "%.4f seconds for "  %(end-start)   + fun.__name__
		return result
	return wrapper

# @jit
def F(rhoM, proDict):
	return reduce( lambda x,y:x+y,[ proDict[pm][0] * log(np.trace(np.dot(proDict[pm][1], rhoM))) for pm in proDict])

# F_numba = autojit(F)

# @jit
def Grad_F(rhoM, proDict, dim):
	return reduce ( lambda x,y:x+y,[(proDict[pm][0] * proDict[pm][1] / (np.trace(np.dot(proDict[pm][1], rhoM)))) for pm in  proDict], np.zeros((dim, dim)))

# Grad_F_numba = autojit(Grad_F)	

# @jit
def D(t, rhoM, Grad_Vaule, FRF, FR, RF):
	a_q_temp=1 + 2 * t + t * t *np.trace(FRF)
	temp1 = (2 * ( (FR +RF) / 2 - rhoM) / a_q_temp)
	temp2 = (t * np.trace(FRF) * ((FRF / np.trace(FRF)) - rhoM) / a_q_temp)
	res = temp2 + temp1
	return res
# D_numba = autojit(D)

# @timeDeco
# @jit
def judge_t(t, d, rhoM, proDict, Grad_Vaule, iter_r):
	temp1 = F(rhoM + t * d, proDict)
	temp2 = F(rhoM, proDict)
	
	temp3 = iter_r * t * np.trace(np.dot(Grad_Vaule, d))
	temp = temp1 - temp2 - temp3
	if temp < -1e-5:
		return True
	else:
		return False

# judge_t_numba = autojit(judge_t)


def intiRho_embedding(dim):
	# randomnum = np.random.random(dim)
	randomnum = rng.uniform(0, 1, dim)
	# print randomnum
	diagmat = np.diag(randomnum)

	return diagmat / np.trace(diagmat)

def intiRho_onehot(probs_term, wordset):
	dim = len(wordset)
	vec = np.zeros(dim)
	for word,weight in probs_term.items():
		index= np.where(wordset == word)
		vec[index]=weight
	vec = list(vec)
	res = np.diag(vec)
	return res

# @timeDeco
# @jit
def Judge(rhoM, proDict, t=1, iter_r = 0.685, iter_a = 0.3):
	
	dim = len(proDict[proDict.keys()[0]][1])
	
	count_i = 0
	rhoM0 = rhoM.copy()
	Grad_Vaule = Grad_F(rhoM, proDict, dim)
	FRF = np.dot(np.dot(Grad_Vaule, rhoM), Grad_Vaule)
	FR = np.dot(Grad_Vaule, rhoM)
	RF = np.dot(rhoM, Grad_Vaule)
	while (np.fabs(np.linalg.norm(rhoM) - np.linalg.norm(FRF)) >2*1e-5) :
		
		t = max(1, t)
		iter_D = D(t, rhoM, Grad_Vaule, FRF, FR, RF)
		count_j = 0
		while judge_t(t, iter_D, rhoM, proDict, Grad_Vaule, iter_r):
			ttt = np.trace(np.dot(Grad_Vaule, iter_D))
			# print ttt

			if ttt <= 0:
				break

			t *= iter_a
			
			iter_D = D(t, rhoM, Grad_Vaule, FRF, FR, RF)
			count_j += 1
			if count_j > 100:
				break
		rhoM += t * iter_D
		Grad_Vaule = Grad_F(rhoM, proDict, dim)
		FRF = np.dot(np.dot(Grad_Vaule, rhoM), Grad_Vaule)
		FR = np.dot(Grad_Vaule, rhoM)
		RF = np.dot(rhoM, Grad_Vaule)

		count_i += 1
		if count_i >100 :
			# print '(((('
			# print count_i
			print 'no convergence'
			break

	return rhoM


tmax = 1


@timeDeco
def test():
	array_temp = np.zeros(6)
	array0 = array_temp.copy()
	array0[0] = 1
	array0 = np.outer(array0, array0)
	array1 = array_temp.copy()
	array1[1] = 1
	array1 = np.outer(array1, array1)
	array2 = array_temp.copy()
	array2[2] = 1
	array2 = np.outer(array2, array2)

	input_dict = {'we' : [0.2, array0], ' ' : [0.4, array1], 'are' : [0.4, array2]}
	rhoM0 = np.array([[0.2, 0, 0, 0, 0, 0], [0, 0.1, 0, 0, 0, 0], [0, 0, 0.7, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
	
	rhoM0=Judge( rhoM0, input_dict)
	# print
	
	print rhoM0



if __name__ == '__main__':
	test()


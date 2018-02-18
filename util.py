import numpy as np 
import pandas as pd 
import cPickle as Pickle
from PIL import Image
import os, sys
'''def vec_all_same(y):
	temp = y[0]
	for i in y:
		if i != temp:
			return False
	return True
def data_iter(qidset, questionset, answerset, labelset):
    pre_qid = -10000
    count = 0
    q_list = []
    a_list = []
    l_list = []
    for index, qaid in enumerate(qidset):
    	qid = qaid[0]
        if qid == pre_qid:
            q_list.append(questionset[index])
            a_list.append(answerset[index])
            l_list.append(labelset[index])
        else:
            if q_list == []:
            	q_list.append(questionset[index])
                a_list.append(answerset[index])
                l_list.append(labelset[index])
                pre_qid = qid
                continue
            else:
                yield np.array(q_list), np.array(a_list), np.array(l_list)
                q_list = []
                a_list = []
                l_list = []
                q_list.append(questionset[index])
                a_list.append(answerset[index])
                l_list.append(labelset[index])
        pre_qid = qid
    yield np.array(q_list), np.array(a_list), np.array(l_list)

def sample_pad(x1, x2, y, batch_size):
	if len(x1) > batch_size:
		tail = batch_size / 2
		head = batch_size - tail
		x1 = np.concatenate([x1[ : head], x1[-tail : ]])
		x2 = np.concatenate([x2[ : head], x2[-tail : ]])
		y = np.concatenate([y[ : head], y[-tail : ]])
	while len(x1) < batch_size:
		x1 = np.concatenate([x1, x1])
		x2 = np.concatenate([x2, x2])
		y = np.concatenate([y, y])
	x1 = x1[:batch_size]
	x2 = x2[:batch_size]
	y = y[:batch_size]
	if vec_all_same(y):
		y[-1] = 1 - y[-1]
		x1[-1] = np.zeros((x1[-1].shape))
		x2[-1] = np.zeros((x2[-1].shape))
	return x1, x2, y
def score_regain(score, initlen):
	score = list(score)
	nowlen = len(score)
	regain_score = []
	if initlen <= nowlen:
		regain_score = score[ : initlen]
	else:
		tail_len = nowlen / 2
		head_len = nowlen - tail_len
		pad_len = initlen - nowlen
		temp_score = []
		while len(temp_score) < pad_len:
			temp_score += score
		temp_score = temp_score[ : pad_len]
		regain_score = score[ : head_len] + temp_score + score[-tail_len : ]
	return regain_score


def load_bin_vec(fname):
	"""
	Loads 300x1 word vecs from Google (Mikolov) word2vec
	"""
	print fname
	word_vecs = {}
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * layer1_size
		print 'vocab_size, layer1_size', vocab_size, layer1_size
		for i, line in enumerate(xrange(vocab_size)):
			if i % 100000 == 0:
				print '.',
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)
			word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
		print "done"
		return word_vecs

def slide_window(sen, worddict={}, step = 7):
	# sen_list = [i for i in sen.split() if i in worddict.keys()]
	sen_list =  sen.split()
	res = []
	for i, word in enumerate(sen_list):
		try:
			den_term = sen_list[i: i + step]
		except:
			den_term = sen_list[i:]
		dency_term = [i for i in den_term if i in worddict.keys()]
		if len(dency_term) > 1:
			res.append(dency_term)
	return res

def density_iter(density_file, batchs, batch_size):
	density_data = open(density_file, 'rb')
	for batch in range(batchs):
		res_q = []
		res_a = []
		res_l = [] 
		matrix_shape = 0
		for sample in range(batch_size):
			try:
				_, _, quesitonRho, answerRho, label= Pickle.load(density_data)
				if matrix_shape == 0:
					matrix_shape = quesitonRho.shape
			except:
				# break
				density_data.close()
				density_data = open(density_file, 'rb')
				_, _, quesitonRho, answerRho, label= Pickle.load(density_data)
			res_q.append(quesitonRho)
			res_a.append(answerRho)
			# res_l.append(label)
		if batch == (batchs - 1):
			density_data.close()
		yield np.concatenate(res_q).reshape(batch_size, matrix_shape[0], matrix_shape[1]), \
		np.concatenate(res_a).reshape(batch_size, matrix_shape[0], matrix_shape[1])


def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8), mode = 'P')
    # new_im = Image.fromarray(data.astype(np.uint8))
    L, W = new_im.size
    new_im = new_im.resize((10 * L, 10 * W))
    return new_im
'''
def batch_pad(datasets, batchsize):
	batch_num = int(len(datasets[0]) / batchsize)
	extra_num = len(datasets[0]) - batch_num * batchsize
	if extra_num > 0:
		pad_num = batchsize - extra_num
		for index, dataset in enumerate(datasets):
			datasets[index] += dataset[:pad_num]
		return batch_num + 1
	return batch_num
def briany_test_file(df_test, mode = 'test'):
	N = len(df_test)
	nnet_outdir = 'exp.out/' + mode
	if not os.path.exists(nnet_outdir):
		os.makedirs(nnet_outdir)

	df_submission = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
	df_submission['qid'] = df_test['qid']
	df_submission['iter'] = 0
	df_submission['docno'] = np.arange(N)
	df_submission['rank'] = 0
	df_submission['sim'] = df_test['score']
	df_submission['run_id'] = 'nnet'
	df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

	df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
	df_gold['qid'] = df_test['qid']
	df_gold['iter'] = 0
	df_gold['docno'] = np.arange(N)
	df_gold['rel'] = df_test['flag']
	df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')


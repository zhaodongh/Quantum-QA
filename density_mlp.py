#coding=utf8
from __future__ import division
import numpy as np
import pandas as pd
import cPickle as Pickle
from scipy.sparse import coo_matrix
import os, time, sys
from sklearn import linear_model, metrics
from PIL import Image
# import matplotlib.pyplot as plt 
import ConfigParser
from sklearn.utils import shuffle


import theano
import theano.tensor as T 
from theano.tensor.nnet import conv
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample

from util import batch_pad, data_iter, score_regain, MatrixToImage, briany_test_file
from cnn_net import HiddenLayer, LogisticRegression, cnn_net, share_x, share_y, Attention_Matrix, MaxMatrix_SMatrix, \
eaual_pad, Cosine_Layer, Density_Dot, Density, Trace_Inner, Get_Diag
from evaluation import evaluation_plus, mrr_metric, map_metric

cf = ConfigParser.ConfigParser()
cf.read('config.conf')
dirs, file_path, train_files, dev_files, test_files, gobal_var, _, _, _, model_arg = cf.sections()

log_dir = cf.get(dirs, 'log_dir')
image_dir = cf.get(dirs, 'image_dir')

vocab_fname = cf.get(file_path, 'vocab_fname')
train_inter_fname = cf.get(file_path, 'train_inter_fname')
test_inter_fname = cf.get(file_path, 'test_inter_fname')
train_fname = cf.get(file_path, 'train_fname')
test_fname = cf.get(file_path, 'test_fname')

q_lens = cf.getint(gobal_var, 'qlen')
a_lens = cf.getint(gobal_var, 'alen')
row_equal = cf.getboolean(gobal_var, 'rowequal')
col_equal = cf.getboolean(gobal_var, 'colequal')

train_result_fname = cf.get(train_files, 'result')

test_result_fname = cf.get(test_files, 'result')

# np.set_printoptions(threshold='nan')

train_result_fname += str(time.time())
test_result_fname += str(time.time())

if __name__ == '__main__':

	learning_rate = cf.getfloat(model_arg, 'learning_rate')
	learning_rate = 0.01
	batch_size = cf.getint(model_arg, 'batch_size')
	filter_row = cf.getint(model_arg, 'filter_row')
	filter_num = cf.getint(model_arg, 'filter_num')
	reg_rate = cf.getfloat(model_arg, 'reg_rate')

	try:
		learning_rate = float(sys.argv[1])
		batch_size = int(sys.argv[2])
		filter_num = int(sys.argv[3])
		filter_row = int(sys.argv[4])
		reg_rate = float(sys.argv[5])
		log_file = sys.argv[6]
		log_file = open(log_file, 'a')
	except:
		log_file = os.path.join(log_dir, 'single_ovelap' + str(time.time()))
		log_file = open(log_file, 'w')

	df_train= pd.read_csv(train_fname, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)
	df_test= pd.read_csv(test_fname, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)

	# df_test.groupby("question").apply(lambda group: len(group))
	
	
	train_max = df_train.groupby('qid').count().max()[0]
	test_max = df_test.groupby('qid').count().max()[0]
	print 'learning_rate is: ', learning_rate 
	print 'batch_size is: ', batch_size
	print 'step is: ', filter_row
	print 'feature map is: ', filter_num


	train_inter = Pickle.load(open(train_inter_fname, 'rb'))

	test_inter = Pickle.load(open(test_inter_fname, 'rb'))

	embeddings = Pickle.load(open(vocab_fname, 'rb'))

	embedding_ndim = len(embeddings[0])

	filter_col = filter_row

	if row_equal:
		eaual_pad(train_inter[1], filter_row)
		eaual_pad(train_inter[2], filter_row)
		eaual_pad(test_inter[1], filter_row)
		eaual_pad(test_inter[2], filter_row)
		q_lens += filter_row - 1
		a_lens += filter_row - 1

	input_shape = (embedding_ndim, embedding_ndim)
	
	train_sample_num = len(train_inter[3])
	test_sample_num = len(test_inter[3])

	
	# print q_lens
	# print a_lens
	train_batch_num = batch_pad(train_inter, batch_size)
	test_batch_num = batch_pad(test_inter, batch_size)
	
	
						
	share_list = [train_inter[1], train_inter[2], train_inter[4], train_inter[5], train_inter[6], \
	test_inter[1], test_inter[2], test_inter[4], test_inter[5], test_inter[6]]

	train_inter[1], train_inter[2], train_inter[4], train_inter[5], train_inter[6], \
	test_inter[1], test_inter[2], test_inter[4], test_inter[5], test_inter[6] = share_x(share_list)

	train_inter[7], train_inter[8], test_inter[7], test_inter[8] = \
	share_x([train_inter[7], train_inter[8], test_inter[7], test_inter[8]])

	train_y = share_y(train_inter[3])
	test_y = share_y(test_inter[3])

	print 'data sets have been shared'

	
	result = []

	channel_num = 1

	image_shape = (batch_size, channel_num, input_shape[0], input_shape[1])

	filter_shape = (filter_num, channel_num, filter_row, filter_col)

	conv_num = int(input_shape[0] - filter_row + 1) * int(input_shape[1] - filter_col + 1)

	class_num = 2

	rng = np.random.RandomState(23455)

	index = T.lscalar('index')
	x_q = T.matrix('x_q')
	x_a = T.matrix('x_a')   
	y = T.ivector('y')
	len_q = T.vector('len_q')
	len_a = T.vector('len_a')
	overlap = T.vector('overlap')
	idf_overlap = T.vector('idf_overlap')
	densim = T.vector('densim')
	pos = T.lscalar('pos')
	neg = T.lscalar('neg')

	Embeddings = theano.shared(value = embeddings, name = 'Embeddings', borrow=True)
	layer0_input_q = Embeddings[T.cast(x_q.flatten(),dtype="int32")].\
	reshape((batch_size, q_lens, embedding_ndim))
	layer0_input_a = Embeddings[T.cast(x_a.flatten(),dtype="int32")].\
	reshape((batch_size, a_lens, embedding_ndim))

	layer0_q = Density(rng, layer0_input_q, len_q, q_lens)
	layer0_a = Density(rng, layer0_input_a, len_a, a_lens)

	density_dot = Density_Dot(rng, layer0_q.output, layer0_a.output).output
	layer1_input = density_dot.reshape(image_shape)

	layer1 = cnn_net(rng, input = layer1_input, filter_shape = filter_shape, \
		image_shape = image_shape)

	# layer1_out = layer1.output.flatten(3)
	trace_inner_feature = Trace_Inner(density_dot).output.reshape((batch_size, 1))
	diag_feature = Get_Diag(density_dot).output.reshape((batch_size, embedding_ndim))
	

	overlap_feature = overlap.reshape((batch_size, 1))
	idf_overlap_feature = idf_overlap.reshape((batch_size, 1))
	# densim_feature = densim.reshape((batch_size, 1))
	len_q_feature = len_q.reshape((batch_size, 1))
	len_a_feature = len_a.reshape((batch_size, 1))

	layer2_in_q = layer1.output.max(axis = 3).flatten(2)
	layer2_in_a = layer1.output.max(axis = 2).flatten(2)

	# Logistic_input = T.concatenate([layer2_in_q, layer2_in_a], axis = 1)

	# layer2 = LogisticRegression(Logistic_input, Logistic_in_num, class_num)

	# Logistic_input = T.concatenate([layer2_in_q, layer2_in_a, idf_overlap_feature], axis = 1)
	# Logistic_input = T.concatenate([layer2_in_q, layer2_in_a, overlap_feature, \
	# 	idf_overlap_feature, densim_feature, len_a_feature\
	# 	], axis = 1).flatten(2)
	# layer2 = LogisticRegression(Logistic_input, Logistic_in_num + 4, class_num)
	Logistic_in_num = 2 * int(input_shape[0] - filter_row + 1) * filter_num

	Logistic_input = T.concatenate([trace_inner_feature, diag_feature], axis = 1).flatten(2)

	layer2 = LogisticRegression(Logistic_input, embedding_ndim + 1, class_num)

	# Logistic_input = T.concatenate([trace_inner_feature, diag_feature, overlap_feature, \
	# 	idf_overlap_feature, len_q_feature, len_a_feature\
	# 	], axis = 1).flatten(2)

	# layer2 = LogisticRegression(Logistic_input, embedding_ndim + 5, class_num)

	
	# layer2_in = layer1.output.max(axis = 3)
	# layer2_in = layer2_in.max(axis = 2).flatten(2)
	# Logistic_in_num = filter_num

	# # Logistic_input = T.concatenate([layer2_in], axis = 1)

	# # layer2 = LogisticRegression(Logistic_input, Logistic_in_num, class_num)

	# Logistic_input = T.concatenate([layer2_in, overlap_feature, \
	# 	idf_overlap_feature, densim_feature, len_q_feature, len_a_feature\
	# 	], axis = 1).flatten(2)

	# layer2 = LogisticRegression(Logistic_input, Logistic_in_num + 5, class_num)

	# params = layer0_q.params + layer0_a.params + layer1.params + layer2.params + [Embeddings]
	params = layer0_q.params + layer0_a.params + layer2.params + [Embeddings]
	# params = layer0_q.params + layer0_a.params + layer2.params
	cost = layer2.negative_log_likelihood(y)
	# cost = layer2.negative_log_likelihood(y, layer0_q.params[0], layer0_a.params[0], \
	# 	layer1.params[0], layer2.params[0])



	grads = T.grad(cost, params)
	updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

	given_train = {x_q: train_inter[1][index * batch_size : (index + 1) * batch_size], \
	x_a: train_inter[2][index * batch_size : (index + 1) * batch_size], \
	overlap: train_inter[4][index * batch_size : (index + 1) * batch_size], \
	idf_overlap: train_inter[8][index * batch_size : (index + 1) * batch_size], \
	# densim: train_inter[7][index * batch_size : (index + 1) * batch_size], \
	len_q: train_inter[5][index * batch_size : (index + 1) * batch_size], \
	len_a: train_inter[6][index * batch_size : (index + 1) * batch_size], \
	y: train_y[index * batch_size : (index + 1) * batch_size]}
	given_test = {x_q: test_inter[1][index * batch_size : (index + 1) * batch_size], \
	x_a: test_inter[2][index * batch_size : (index + 1) * batch_size], \
	overlap: test_inter[4][index * batch_size : (index + 1) * batch_size], \
	idf_overlap: test_inter[8][index * batch_size : (index + 1) * batch_size], \
	# densim: test_inter[7][index * batch_size : (index + 1) * batch_size], \
	len_q: test_inter[5][index * batch_size : (index + 1) * batch_size], \
	len_a: test_inter[6][index * batch_size : (index + 1) * batch_size], \
	y: test_y[index * batch_size : (index + 1) * batch_size]}


	train_model = theano.function([index], cost, updates = updates, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	train_score_model = theano.function([index], layer2.y_score, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	test_model = theano.function([index], layer2.y_score, givens = given_test, on_unused_input='ignore', allow_input_downcast=True)
	test_cost_model = theano.function([index], cost, givens = given_test, on_unused_input='ignore', allow_input_downcast=True)
	train_model_hahah = theano.function([index],layer1.output.shape, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	train_model_hahah1 = theano.function([index], Logistic_input.shape, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	train_model_hahah2 = theano.function([],layer0_q.W, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	train_model_hahah3 = theano.function([],layer0_a.W, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)

	model_layer0_q_weight = theano.function([],layer0_q.W, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	model_layer0_a_weight = theano.function([],layer0_a.W, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	model_layer1_weight = theano.function([],layer1.W, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	model_layer2_weight = theano.function([],layer2.W, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)


	train_model_loss = theano.function([index], cost, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	test_model_loss = theano.function([index], cost, givens = given_test, on_unused_input='ignore', allow_input_downcast=True)

	itera_number = 25

	

	
	print "begin to train"
	best_map = [0., 0., 0.]
	best_mrr = [0., 0., 0.]
	stop_flag = 0
	for itera_i in range(itera_number):
		if stop_flag > 5:
			break
		print 'The ', itera_i, 'epcho begin'
		count_i = 0
		train_score = []
		for i in shuffle(range(train_batch_num), random_state = 121):
			# print train_model_hahah(i)
			# print train_model_hahah1(i)
			# exit()

			train_model(i)
		
			if count_i % 10 == 0:
				print count_i, 'train have been trained, loss is ', train_model_loss(i)
			count_i += 1

		count_i = 0
		
		
		for i in range(train_batch_num):
			score = train_score_model(i)
			for j in score:
				train_score.append(j[1])
				# train_score.append(j)
			# train_score.append(sim_score)
			if count_i % 10 == 0:
				print count_i, 'train have been tested, loss is ', train_model_loss(i)

			count_i += 1

		train_score = train_score[:train_sample_num]


		df_train['score'] = train_score
		df_train.to_csv('train.result.csv', header=None,sep="\t",index=False, names=['qid','aid',"question","answer","flag",'score'],quoting =3)
		train_flag = df_train['flag']
		print metrics.roc_auc_score(train_flag, train_score) * 100
		print df_train.groupby('question').apply(mrr_metric).mean()
		print df_train.groupby('question').apply(map_metric).mean()

		test_score = []
		count_i = 0
		for i in range(test_batch_num):
			score = test_model(i)
			for j in score:
				test_score.append(j[1])
			if count_i % 10 == 0:
				print count_i, 'test have been tested, loss is ', test_model_loss(i)
			count_i += 1
		test_score = test_score[:test_sample_num]
		
		df_test['score'] = test_score
		df_test.to_csv('test.result.csv', header=None,sep="\t",index=False, names=['qid','aid',"question","answer","flag",'score'],quoting =3)
		# print test_score
		test_flag = df_test['flag']
		score_record = 'socre.txt'
		out = open(score_record, 'w')
		for score in test_score:
			out.write(str(score) + '\n')
		out.close()
		print metrics.roc_auc_score(test_flag, test_score) * 100
		test_mrr = df_test.groupby('question').apply(mrr_metric).mean()
		test_map = df_test.groupby('question').apply(map_metric).mean()
		print test_mrr
		print test_map
		stop_flag += 1
		if test_mrr > best_mrr[0]:
			best_mrr = [test_mrr, test_map, itera_i]
			stop_flag = 0
		if test_map > best_map[1]:
			stop_flag = 0
			best_map = [test_mrr, test_map, itera_i]
			briany_test_file(df_test)
			# briany_test_file(df_train, 'train')


	log_file.write('learning_rate ' + str(learning_rate) + ', ')
	log_file.write('batch_size ' + str(batch_size) + ', ')
	log_file.write('filter_num ' + str(filter_num) + ', ')
	log_file.write('filter_row ' + str(filter_row) + ', ')
	log_file.write('reg_rate ' + str(reg_rate) + ' :\n')
	print 'end train'
	print 'best mrr is : ', best_mrr
	print 'best map is : ', best_map
	log_file.write('best mrr is : ')
	for num in best_mrr:
		log_file.write(str(num) + '\t')
	log_file.write('\n')
	log_file.write('best map is : ')
	for num in best_map:
		log_file.write(str(num) + '\t')
	log_file.write('\n\n')
	log_file.close()
	# layer0q_weight.close()
	# layer0a_weight.close()
	# layer1_weight.close()
	# layer2_weight.close()
	# classer_weight.close()	

import numpy as np 
import cPickle as Pickle 
import os
import pandas as pd 
from time import time
from alphabet import Alphabet
from nltk.corpus import stopwords
import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read('config.conf')
dirs, file_path, train_files, dev_files, test_files, gobal_var, _, _, _, _ = cf.sections()

embedding_fname = cf.get(file_path, 'embedding_fname')
vocab_fname = cf.get(file_path, 'vocab_fname')
index_vocab_fname = cf.get(file_path, 'index_vocab_fname')
train_inter_fname = cf.get(file_path, 'train_inter_fname')
test_inter_fname = cf.get(file_path, 'test_inter_fname')
idf_fname = cf.get(file_path, 'idf_fname')

q_lens = cf.getint(gobal_var, 'qlen')
a_lens = cf.getint(gobal_var, 'alen')

train_fname = cf.get(file_path, 'train_fname')
train_qindex_fname = cf.get(train_files, 'question_index')
train_aindex_fname = cf.get(train_files, 'answer_index')
train_label_fname = cf.get(train_files, 'label')
train_qaid_fname = cf.get(train_files, 'qaid')
train_overlap_value_fname = cf.get(train_files, 'overlap_value')
train_density_sim_fname = cf.get(train_files, 'density_sim')
train_qlen_fname = cf.get(train_files, 'qlen')
train_alen_fname = cf.get(train_files, 'alen')


test_fname = cf.get(file_path, 'test_fname')
test_qindex_fname = cf.get(test_files, 'question_index')
test_aindex_fname = cf.get(test_files, 'answer_index')
test_label_fname = cf.get(test_files, 'label')
test_qaid_fname = cf.get(test_files, 'qaid')
test_overlap_value_fname = cf.get(test_files, 'overlap_value')
test_density_sim_fname = cf.get(test_files, 'density_sim')
test_qlen_fname = cf.get(test_files, 'qlen')
test_alen_fname = cf.get(test_files, 'alen')

word_count = [0]
random_word_count = [0]
UNKNOWN_WORD_IDX_0 = 0


rng = np.random.RandomState(23455)

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

def add_to_vocab(data, alphabet):
	for sentence in data:
		for token in sentence.split():
			alphabet.add(token)

def sentence_index(sen, alphabet, input_lens):
	sen = sen.split()
	sen_index = []
	for word in sen:
		sen_index.append(alphabet[word])

	# sen_index = sen_index + [0] * (input_lens - len(sen_index))
	sen_index = sen_index[:input_lens]

	while len(sen_index) < input_lens:
		sen_index += sen_index[:(input_lens - len(sen_index))]

	return np.array(sen_index), len(sen)

def Sentence_indece(crous, alphabet,):
	qids = crous['qid']
	aids = crous['aid']
	questions = crous['question']
	answers = crous['answer']
	labels = crous['flag']
	question_indece = []
	answer_indece = []
	qlen_list = []
	alen_list = []
	for question in questions:
		question_index, question_len = sentence_index(question, alphabet, q_lens)
		question_indece.append(question_index)
		qlen_list.append(question_len)
	for answer in answers:
		answer_index, answer_len = sentence_index(answer, alphabet, a_lens)
		answer_indece.append(answer_index)
		alen_list.append(answer_len)
	labels_list = list(labels)
	qids_list = list(qids)
	aids_list = list(aids)
	qaids_list = []
	for i, j in zip(qids_list, aids_list):
		qaids_list.append([i, j])
	return qaids_list, question_indece, answer_indece, labels_list, qlen_list, alen_list

def compute_overlap(qindex, aindex, q_lens, a_lens):
	qset = set(qindex)
	aset = set(aindex)
	overlap = qset.intersection(aset)
	overlap_lens = max(q_lens, a_lens)

	qoverlap = np.zeros(overlap_lens)
	for i, q in enumerate(qindex):
		value = 0
		if q in overlap:
			value = 1
		qoverlap[i] = value

	aoverlap = np.zeros(overlap_lens)
	for i, a in enumerate(aindex):
		value = 0
		if a in overlap:
			value = 1
		aoverlap[i] = value
	return qoverlap, aoverlap

def Compute_Overlaps(qindece, aindece, q_lens, a_lens):
	qoverlaps = []
	aoverlaps = []
	for qindex, aindex in zip(qindece, aindece):
		qoverlap, aoverlap = compute_overlap(qindex, aindex, q_lens, a_lens) 
		qoverlaps.append(qoverlap)
		aoverlaps.append(aoverlap)

	return qoverlaps, aoverlaps

def Compute_Overlao_Values(crous, stoplist = []):
	questions = crous['question']
	answers = crous['answer']
	overlaps = []
	for question, answer in zip(questions, answers):
		question = question.split()
		answer = answer.split()
		qindex = [q for q in question if q not in stoplist]
		# print len(qindex)
		aindex = [a for a in answer if a not in stoplist]
		qset = set(qindex)
		# overlap = [a for a in aindex if a in qset]
		aset = set(aindex)
		# overlap = [q for q in qindex if q in aset]
		overlap = qset.intersection(aset)
		overlaps.append(len(overlap))
		# overlaps.append(float(len(overlap) / (len(qset) + len(aset))))
	return overlaps

def Compute_Overlao_Values_idf(crous, stoplist = [], idf_dict = {}):
	questions = crous['question']
	answers = crous['answer']
	overlaps = []
	for question, answer in zip(questions, answers):
		question = question.split()
		answer = answer.split()
		qindex = [q for q in question if q not in stoplist]
		# print len(qindex)
		aindex = [a for a in answer if a not in stoplist]
		qset = set(qindex)
		# overlap = [a for a in aindex if a in qset]
		aset = set(aindex)
		# overlap = [q for q in qindex if q in aset]
		overlap = qset.intersection(aset)
		count = 0
		for i in overlap:
			try:
				count += idf_dict[i]
			except:
				print i
		overlaps.append(count)
	temp = np.array(overlaps, dtype='float32')
	overlaps = list(temp)
	return overlaps


if __name__ == '__main__':
	wiki_dict = load_bin_vec(embedding_fname)
	ndim = len(wiki_dict[wiki_dict.keys()[0]])
	df_train= pd.read_csv(train_fname, header=None,sep="\t",names=["qid",'aid',"question","answer","flag"],quoting =3)
	df_test= pd.read_csv(test_fname, header=None,sep="\t",names=["qid",'aid',"question","answer","flag"],quoting =3)

	df_train['question'] = df_train['question'].str.lower()
	df_train['answer'] = df_train['answer'].str.lower()
	df_test['question'] = df_test['question'].str.lower()
	df_test['answer'] = df_test['answer'].str.lower()
	# stopwords = stopwords.words('english')
	stopwords = []
	print 'load idf dict'
	ex_idf_dict = Pickle.load(open(idf_fname, 'rb'))
	print 'idf dict has been loaded'

	alphabet = Alphabet(start_feature_id=0)
	alphabet.add('UNKNOWN_WORD_IDX_0')

	vocab_dict = {}
	idf_dict = {}

	for crous in [df_train, df_test]:
		add_to_vocab(crous['question'], alphabet)
		add_to_vocab(crous['answer'], alphabet)

	print alphabet.fid
	temp_vec = 0
	vocab_array = np.zeros((alphabet.fid, ndim), dtype = 'float32')
	for index in alphabet.keys():
		vec = wiki_dict.get(index, None)
		if vec is None:
			vec = rng.uniform(-0.25, 0.25, ndim)
			vec = list(vec)
			vec = np.array(vec, dtype = 'float32')
			random_word_count[0] += 1
		if type(vec[0]) is not np.float32:
			print type(vec[0])
		if alphabet[index] == 0:
			vec = np.zeros(ndim)
		# vocab_array[alphabet[index]] = (vec + 3)/6
		# vec -= vec.sum()/ len(vec)
		temp_vec += vec
		vocab_array[alphabet[index]] = vec
	temp_vec /= len(vocab_array)
	for index, _ in enumerate(vocab_array):
		vocab_array[index] -= temp_vec
	
	for index in alphabet.keys():
		idf = ex_idf_dict.get(index, None)
		if idf is None:
			idf = 0
		idf_dict[index] = idf

	Pickle.dump(alphabet, open(index_vocab_fname, 'wb'))
	Pickle.dump(vocab_array, open(vocab_fname, 'wb'))
	print alphabet.fid

	

	print 'train data begin to pro'
	train_qaid, train_qindex, train_aindex, train_label, train_qlen, train_alen \
	= Sentence_indece(df_train, alphabet)
	# train_qoverlap, train_aoverlap =  Compute_Overlaps(train_qindex, train_aindex, q_lens, a_lens)
	train_overlap_value =  Compute_Overlao_Values(df_train, stopwords)
	train_overlap_value_idf = Compute_Overlao_Values_idf(df_train, stopwords, idf_dict)
	# train_overlap_value =  Compute_Overlao_Values(train_qindex, train_aindex)
	print 'train data has been proed'
	print 'test data begin to pro'
	test_qaid, test_qindex, test_aindex, test_label, test_qlen, test_alen \
	= Sentence_indece(df_test, alphabet)
	# test_qoverlap, test_aoverlap = Compute_Overlaps(test_qindex, test_aindex, q_lens, a_lens)
	test_overlap_value =  Compute_Overlao_Values(df_test, stopwords)
	test_overlap_value_idf = Compute_Overlao_Values_idf(df_test, stopwords, idf_dict)
	# test_overlap_value =  Compute_Overlao_Values(test_qindex, test_aindex)
	print 'test data has been proed'

	Pickle.dump(train_qaid, open(train_qaid_fname, 'wb'))
	Pickle.dump(train_qindex, open(train_qindex_fname, 'wb'))
	Pickle.dump(train_aindex, open(train_aindex_fname, 'wb'))
	Pickle.dump(train_label, open(train_label_fname, 'wb'))
	# Pickle.dump(train_qoverlap, open(train_qoverlap_file, 'wb'))
	# Pickle.dump(train_aoverlap, open(train_aoverlap_file, 'wb'))
	Pickle.dump(train_overlap_value, open(train_overlap_value_fname, 'wb'))
	Pickle.dump(train_qlen, open(train_qlen_fname, 'wb'))
	Pickle.dump(train_alen, open(train_alen_fname, 'wb'))
	train_density_sim = []
	train_inter = [train_qaid, train_qindex, train_aindex, train_label, train_overlap_value, \
	train_qlen, train_alen, train_density_sim, train_overlap_value_idf]
	Pickle.dump(train_inter, open(train_inter_fname, 'wb'))


	Pickle.dump(test_qaid, open(test_qaid_fname, 'wb'))
	Pickle.dump(test_qindex, open(test_qindex_fname, 'wb'))
	Pickle.dump(test_aindex, open(test_aindex_fname, 'wb'))
	Pickle.dump(test_label, open(test_label_fname, 'wb'))
	# Pickle.dump(test_qoverlap, open(test_qoverlap_file, 'wb'))
	# Pickle.dump(test_aoverlap, open(test_aoverlap_file, 'wb'))
	Pickle.dump(test_overlap_value, open(test_overlap_value_fname, 'wb'))
	Pickle.dump(test_qlen, open(test_qlen_fname, 'wb'))
	Pickle.dump(test_alen, open(test_alen_fname, 'wb'))
	test_density_sim = []
	test_inter = [test_qaid, test_qindex, test_aindex, test_label, test_overlap_value, \
	test_qlen, test_alen, test_density_sim, test_overlap_value_idf]
	Pickle.dump(test_inter, open(test_inter_fname, 'wb'))

	


import ConfigParser
import os
import sys
from os.path import join as Join 

cf = ConfigParser.ConfigParser()

sec1 = 'path'
sec2 = 'filename'
sec3 = 'train_filename'
sec4 = 'dev_filename'
sec5 = 'test_filename'
sec6 = 'gobal_var'
sec7 = 'trec_arg'
sec8 = 'wiki_arg'
sec10 = 'trec-all_arg'

cf.add_section(sec1)
cf.set(sec1, 'data_dir', 'data')
cf.set(sec1, 'crous', 'trec-all') #'trec','wiki', 'trec-all'
cf.set(sec1, 'dataset', 'data_set')
cf.set(sec1, 'inter', 'inter_result')
cf.set(sec1, 'final', 'final_result')
cf.set(sec1, 'embedding_dir', 'embedding_dir')
cf.set(sec1, 'statc_fatures', 'statc_fatures')

data_dir = cf.get(sec1, 'data_dir')
crous = cf.get(sec1, 'crous')
dataset = Join(data_dir, crous, cf.get(sec1, 'dataset'))
inter = Join(data_dir, crous, cf.get(sec1, 'inter'))
final = Join(data_dir, crous, cf.get(sec1, 'final'))
embedding_dir = Join(data_dir, cf.get(sec1, 'embedding_dir'))
statc_fatures = Join(data_dir, cf.get(sec1, 'statc_fatures'))
for path in [dataset, inter, final, embedding_dir, statc_fatures]:
	if not os.path.exists(path):
		os.makedirs(path)
cf.set(sec1, 'log_dir', Join(data_dir, crous, 'log'))
log_dir = cf.get(sec1, 'log_dir')
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
cf.set(sec1, 'image_dir', Join(data_dir, crous, 'image'))

cf.add_section(sec2)
cf.set(sec2, 'train_fname',  Join(dataset, 'train'))
# cf.set(sec2, 'train_fname',  Join(dataset, 'train_clean'))
cf.set(sec2, 'dev_fname', Join(dataset, 'dev'))
cf.set(sec2, 'test_fname', Join(dataset, 'test'))
# cf.set(sec2, 'test_fname', Join(dataset, 'test_clean'))
cf.set(sec2, 'embedding_fname', Join(embedding_dir, 'aquaint+wiki.txt.gz.ndim=50.bin'))
cf.set(sec2, 'idf_fname', Join(statc_fatures, 'idf.pkl'))
cf.set(sec2, 'vocab_fname', Join(inter, 'vocab'))
cf.set(sec2, 'index_vocab_fname', Join(inter, 'index_vocab'))
cf.set(sec2, 'train_inter_fname', Join(inter, 'train_inter'))
cf.set(sec2, 'dev_inter_fname', Join(inter, 'dev_inter'))
cf.set(sec2, 'test_inter_fname', Join(inter, 'test_inter'))

train_fname = cf.get(sec2, 'train_fname')

for sec in [sec3, sec4, sec5]:
	cf.add_section(sec)
	tdt_name = sec.split('_')[0]
	tdt_path = Join(inter, tdt_name)
	if not os.path.exists(tdt_path):
		os.makedirs(tdt_path)
	cf.set(sec, 'qaid', Join(tdt_path, 'qaid'))
	cf.set(sec, 'label', Join(tdt_path, 'label'))
	cf.set(sec, 'overlap_value', Join(tdt_path, 'overlap_value'))
	cf.set(sec, 'question_index', Join(tdt_path, 'question_index'))
	cf.set(sec, 'answer_index', Join(tdt_path, 'answer_index'))
	cf.set(sec, 'density_sim', Join(tdt_path, 'density_sim'))
	cf.set(sec, 'qlen', Join(tdt_path, 'qlen'))
	cf.set(sec, 'alen', Join(tdt_path, 'alen'))

	tdt_path = Join(final, tdt_name)
	if not os.path.exists(tdt_path):
		os.makedirs(tdt_path)
	cf.set(sec, 'result', Join(tdt_path, 'result'))


cf.add_section(sec6)
cf.set(sec6, 'qlen', '50')
cf.set(sec6, 'alen', '100')
cf.set(sec6, 'rowequal', True)
cf.set(sec6, 'colequal', True)

cf.add_section(sec7)
cf.set(sec7, 'learning_rate', '0.01')
cf.set(sec7, 'batch_size', '20')
cf.set(sec7, 'filter_row', '40')
cf.set(sec7, 'filter_num', '70')
cf.set(sec7, 'reg_rate', '0.04')

cf.add_section(sec8)
cf.set(sec8, 'learning_rate', '0.02')
cf.set(sec8, 'batch_size', '150')
cf.set(sec8, 'filter_row', '40')
cf.set(sec8, 'filter_num', '150')
cf.set(sec8, 'reg_rate', '0.01')

cf.add_section(sec10)
cf.set(sec10, 'learning_rate', '0.01')
cf.set(sec10, 'batch_size', '100')
cf.set(sec10, 'filter_row', '40')
cf.set(sec10, 'filter_num', '65')
cf.set(sec10, 'reg_rate', '0.001')


sec9 = crous
cf.add_section(sec9)
cf.set(sec9, 'learning_rate', cf.get(sec9 + '_arg', 'learning_rate'))
cf.set(sec9, 'batch_size', cf.get(sec9 + '_arg', 'batch_size'))
cf.set(sec9, 'filter_row', cf.get(sec9 + '_arg', 'filter_row'))
cf.set(sec9, 'filter_num', cf.get(sec9 + '_arg', 'filter_num'))
cf.set(sec9, 'reg_rate', cf.get(sec9 + '_arg', 'reg_rate'))



# print cf.sections()

cf.write(open('config.conf', 'w'))
print (crous)

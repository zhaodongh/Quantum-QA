import sys
import os
import subprocess
import time
import ConfigParser


cf = ConfigParser.ConfigParser()
cf.read('config.conf')
dirs, file_path, train_files, dev_files, test_files, gobal_var, _, _, _, model_arg = cf.sections()

log_dir = cf.get(dirs, 'log_dir')

learning_rate_list = [0.01 * (i + 1) for i in range(20)]
batch_size_list = [120, 130, 140]
# batch_size_list = [65, 70, 75]
# filter_num_list = [10 * (i+5) for i in range(15)]
filter_num_list = [90, 100, 150, 200, 120]
log_file = os.path.join(log_dir, 'learning_rate' + sys.argv[0].split('\\')[-1] + str(time.time()))
learning_rate =0.02
batch_size = 100
filter_raw = 40
reg_rate = 0.01
filter_num = 150


count = 0

print log_file
for batch_size in batch_size_list:
	print 'The ', count, 'excue\n'
	count += 1
	subprocess.call('python main.py %f %d %d %d %f %s'%(learning_rate, batch_size, \
		filter_num, filter_raw, reg_rate, log_file), shell = True)


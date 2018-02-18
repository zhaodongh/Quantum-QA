import sys
import os
import subprocess
import time
import ConfigParser


cf = ConfigParser.ConfigParser()
cf.read('config.conf')
dirs, file_path, train_files, dev_files, test_files, gobal_var, _, _, _, _ = cf.sections()

log_dir = cf.get(dirs, 'log_dir')

learning_rate_list = [0.01, 0.05, 0.1, 0.03, 0.08, 0.3]
batch_size_list = [10 * (i+1) for i in range(10)]
# batch_size_list = [65, 70, 75]
# filter_num_list = [10 * (i+5) for i in range(15)]
filter_num_list = [65, 70, 75]
log_file = os.path.join(log_dir, sys.argv[0].split('\\')[-1] + str(time.time()))
learning_rate =0.01
batch_size = 20
filter_raw = 40
reg_rate = 0
filter_num = 70

# reg_rate_list = [0.015, 0.025, 0.035, 0.045]


count = 0

print log_file
for learning_rate in learning_rate_list:
	print 'The ', count, 'excue\n'
	count += 1
	subprocess.call('python trace_inner_network.py %f %d %d %d %f %s'%(learning_rate, batch_size, \
		filter_num, filter_raw, reg_rate, log_file))


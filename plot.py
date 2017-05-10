import argparse
import PyGnuplot as pg
import torch
import numpy
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--load_paths', nargs='+', help='paths to experiments to plot')
parser.add_argument('--type', default='test', help='test | dis | dis-real | dis-fake | gen')
parser.add_argument('--interval', type=int, default=100, help='window size for averaged discriminator loss')
opt = parser.parse_args()
print(opt)

for k, load_path in enumerate(opt.load_paths):
	logs = []
	log_files = os.listdir(os.path.join(load_path, 'log'))
	for log_file in log_files:
		niter = int(log_file[5:-3])
		log = torch.load(os.path.join(load_path, 'log', log_file))
		if opt.type == 'test':
			loss = log['test_loss']
			logs.append((niter, loss))
		else:
			losses = log['training_loss']
			for i in range(losses.size(0) // opt.interval):
				avg_loss = losses[i * opt.interval : (i + 1) * opt.interval].mean(0)[0]
				if opt.type == 'dis':
					loss = avg_loss[0] + avg_loss[1]
				elif opt.type == 'dis-real':
					loss = avg_loss[0]
				elif opt.type == 'dis-fake':
					loss = avg_loss[1]
				else:
					loss = avg_loss[2]
				logs.append((niter - losses.size(0) + (i + 1) * opt.interval, loss))
	logs.sort()
	n = len(logs)
	x = torch.Tensor(n)
	y = torch.Tensor(n)
	for i in range(n):
		x[i] = logs[i][0]
		y[i] = logs[i][1]
	pg.s([x.numpy(), y.numpy()])
	if k == 0:
		pg.c('plot "tmp.dat" with lines')
	else:
		pg.c('replot "tmp.dat" with lines')
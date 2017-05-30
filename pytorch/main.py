from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import os.path

from model import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',                          required = True,
	help = 'cifar10 | lsun | imagenet | folder | lfw')

parser.add_argument('--lsun_class',                       default = 'bedroom',
	help = 'class of lsun dataset to use')

parser.add_argument('--dataroot',                         required = True,
	help = 'path to dataset')

parser.add_argument('--batch_size',         type = int,   default = 32,
	help = 'input batch size')

parser.add_argument('--image_size',         type = int,   default = -1,
	help = 'image size')

parser.add_argument('--width',              type = int,   default = -1,
	help = 'image width')

parser.add_argument('--height',             type = int,   default = -1,
	help = 'image height')

parser.add_argument('--crop_size',          type = int,   default = -1,
	help = 'crop size before scaling')

parser.add_argument('--crop_width',         type = int,   default = -1,
	help = 'crop width before scaling')

parser.add_argument('--crop_height',        type = int,   default = -1,
	help = 'crop height before scaling')

parser.add_argument('--code_size',          type = int,   default = 128,
	help = 'size of latent code')

parser.add_argument('--nfeature',           type = int,   default = 64,
	help = 'number of features of first conv layer')

parser.add_argument('--nlayer',             type = int,   default = -1,
	help = 'number of down/up conv layers')

parser.add_argument('--norm',                             default = 'none',
	help = 'type of normalization: none | batch | weight | weight-affine')

parser.add_argument('--save_path',                        default = None,
	help = 'path to save generated files')

parser.add_argument('--load_path',                        default = None,
	help = 'load to continue existing experiment')

parser.add_argument('--lr',                 type = float, default = 0.0001,
	help = 'learning rate')

parser.add_argument('--test_interval',      type = int,   default = 1000,
	help = 'how often to test reconstruction')

parser.add_argument('--test_lr',            type = float, default = 0.01,
	help = 'learning rate for reconstruction test')

parser.add_argument('--test_steps',         type = int,   default = 50,
	help = 'number of steps in running reconstruction test')

parser.add_argument('--vis_interval',       type = int,   default = 100,
	help = 'how often to save generated samples')

parser.add_argument('--vis_size',           type = int,   default = 10,
	help = 'size of visualization grid')

parser.add_argument('--vis_row',            type = int,   default = -1,
	help = 'height of visualization grid')

parser.add_argument('--vis_col',            type = int,   default = -1,
	help = 'width of visualization grid')

parser.add_argument('--save_interval',      type = int,   default = 20000,
	help = 'how often to save network')

parser.add_argument('--niter',              type = int,   default = 100000,
	help = 'number of iterations to train')

parser.add_argument('--final_test',         action = 'store_true', default = False,
	help = 'do final test')

parser.add_argument('--ls',                 action = 'store_true', default = False,
	help = 'use LSGAN')

parser.add_argument('--output_scale',       action = 'store_true', default = False,
	help = 'save x*2-1 instead of x when saving image')

parser.add_argument('--net',                              default = 'best',
	help = 'network to load for final test: best | last | <niter>')

opt = parser.parse_args()
print(opt)

transform_list = []

if (opt.crop_height > 0) and (opt.crop_width > 0):
	transform_list.append(transforms.CenterCrop(opt.crop_height, crop_width))
elif opt.crop_size > 0:
	transform_list.append(transforms.CenterCrop(opt.crop_size))

if (opt.height > 0) and (opt.width > 0):
	transform_list.append(transforms.Scale(opt.height, opt.width))
elif opt.image_size > 0:
	transform_list.append(transforms.Scale(opt.image_size))
	transform_list.append(transforms.CenterCrop(opt.image_size))
	opt.height = opt.image_size
	opt.width = opt.image_size
else:
	raise ValueError('must specify valid image size')

transform_list.append(transforms.ToTensor())

if (opt.vis_row <= 0) or (opt.vis_col <= 0):
	opt.vis_row = opt.vis_size
	opt.vis_col = opt.vis_size

if opt.nlayer < 0:
	opt.nlayer = 0
	s = max(opt.width, opt.height)
	while s >= 8:
		s = (s + 1) // 2
		opt.nlayer = opt.nlayer + 1

if opt.dataset == 'cifar10':
	dataset1 = datasets.CIFAR10(root = opt.dataroot, download = True,
		transform = transforms.Compose(transform_list))
	dataset2 = datasets.CIFAR10(root = opt.dataroot, train = False,
		transform = transforms.Compose(transform_list))
	def get_data(k):
		if k < len(dataset1):
			return dataset1[k][0]
		else:
			return dataset2[k - len(dataset1)][0]
else:
	if opt.dataset in ['imagenet', 'folder', 'lfw']:
		dataset = datasets.ImageFolder(root = opt.dataroot,
			transform = transforms.Compose(transform_list))
	elif opt.dataset == 'lsun':
		dataset = datasets.LSUN(db_path = opt.dataroot, classes = [opt.lsun_class + '_train'],
			transform = transforms.Compose(transform_list))
	def get_data(k):
		return dataset[k][0]

data_index = torch.load(os.path.join(opt.dataroot, 'data_index.pt'))
train_index = data_index['train']

if opt.final_test:
	test_index = data_index['final_test']
else:
	test_index = data_index['running_test']

gen = build_generator(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm)
print(gen)
gen.cuda()
testfunc = nn.MSELoss()

if not opt.final_test:
	dis = build_discriminator(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.norm)
	print(dis)
	dis.cuda()
	if opt.ls:
		lossfunc = nn.MSELoss()
	else:
		lossfunc = nn.BCELoss()
	gen_opt = optim.RMSprop(gen.parameters(), lr = opt.lr, eps = 1e-6, alpha = 0.9)
	dis_opt = optim.RMSprop(dis.parameters(), lr = opt.lr, eps = 1e-6, alpha = 0.9)

state = {}

def load_state(path, prefix, gen_only = False):
	gen.load_state_dict(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_gen.pt'.format(prefix))))

	if not gen_only:
		gen_opt.load_state_dict(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_gen_opt.pt'.format(prefix))))
		dis.load_state_dict(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_dis.pt'.format(prefix))))
		dis_opt.load_state_dict(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_dis_opt.pt'.format(prefix))))
		state.update(torch.load(os.path.join(opt.load_path, 'net_archive', '{0}_state.pt'.format(prefix))))

def save_state(path, prefix):
	torch.save(gen.state_dict(), os.path.join(opt.save_path, 'net_archive', '{0}_gen.pt'.format(prefix)))
	torch.save(gen_opt.state_dict(), os.path.join(opt.save_path, 'net_archive', '{0}_gen_opt.pt'.format(prefix)))

	torch.save(dis.state_dict(), os.path.join(opt.save_path, 'net_archive', '{0}_dis.pt'.format(prefix)))
	torch.save(dis_opt.state_dict(), os.path.join(opt.save_path, 'net_archive', '{0}_dis_opt.pt'.format(prefix)))

	state.update({
		'index_shuffle' : index_shuffle,
		'current_iter' : current_iter,
		'best_iter' : best_iter,
		'min_loss' : min_loss,
		'current_sample' : current_sample
	})
	torch.save(state, os.path.join(opt.save_path, 'net_archive', '{0}_state.pt'.format(prefix)))

def visualize(code, filename):
	gen.eval()
	generated = torch.Tensor(code.size(0), 3, opt.height, opt.width)
	for i in range((code.size(0) - 1) // opt.batch_size + 1):
		batch_size = min(opt.batch_size, code.size(0) - i * opt.batch_size)
		batch_code = Variable(code[i * opt.batch_size : i * opt.batch_size + batch_size])
		generated[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(gen(batch_code).data)
	if opt.output_scale:
		torchvision.utils.save_image(generated * 2 - 1, filename, opt.vis_row)
	else:
		torchvision.utils.save_image(generated, filename, opt.vis_row)
	gen.train()

def test():
	test_loss = 0
	for param in gen.parameters():
		param.requires_grad = False
	gen.eval()
	best_code = torch.Tensor(test_index.size(0), opt.code_size).cuda()
	total_batch = (test_index.size(0) - 1) // opt.batch_size + 1

	for i in range(total_batch):
		if opt.final_test:
			print('Testing batch {0} of {1} ...'.format(i + 1, total_batch))
		batch_size = min(opt.batch_size, test_index.size(0) - i * opt.batch_size)
		batch_code = Variable(torch.zeros(batch_size, opt.code_size).cuda())
		batch_code.requires_grad = True

		batch_target = torch.Tensor(batch_size, 3, opt.height, opt.width)
		for j in range(batch_size):
			batch_target[j].copy_(get_data(test_index[i * opt.batch_size + j]))
		batch_target = Variable(batch_target.cuda())

		test_opt = optim.RMSprop([batch_code], lr = opt.test_lr, eps = 1e-6, alpha = 0.9)
		for j in range(opt.test_steps):
			loss = testfunc(gen(batch_code), batch_target)
			loss.backward()
			test_opt.step()
			batch_code.grad.data.zero_()
		best_code[i * opt.batch_size : i * opt.batch_size + batch_size].copy_(batch_code.data)
		
		generated = gen(batch_code)
		loss = testfunc(gen(batch_code), batch_target)
		test_loss = test_loss + loss.data[0] * batch_size
		if opt.final_test:
			print('batch loss = {0}'.format(loss.data[0]))
			sample_rec_pair = torch.Tensor(2, 3, opt.height, opt.width)
			for j in range(batch_size):
				sample_rec_pair[0].copy_(get_data(test_index[i * opt.batch_size + j]))
				sample_rec_pair[1].copy_(generated.data[j])
				if opt.output_scale:
					torchvision.utils.save_image(sample_rec_pair * 2 - 1, os.path.join(opt.load_path, '{0}_test'.format(opt.net), '{0}.png'.format(i * opt.batch_size + j)), 2)
				else:
					torchvision.utils.save_image(sample_rec_pair, os.path.join(opt.load_path, '{0}_test'.format(opt.net), '{0}.png'.format(i * opt.batch_size + j)), 2)

	for param in gen.parameters():
		param.requires_grad = True
	gen.train()
	if not opt.final_test:
		visualize(best_code[0 : min(test_index.size(0), opt.vis_row * opt.vis_col)], os.path.join(opt.save_path, 'running_test', 'test_{0}.jpg'.format(current_iter)))
	test_loss = test_loss / test_index.size(0)
	print('loss = {0}'.format(test_loss))
	return test_loss

if opt.final_test:
	load_state(opt.load_path, opt.net, True)
	if not os.path.exists(os.path.join(opt.load_path, '{0}_test'.format(opt.net))):
		os.mkdir(os.path.join(opt.load_path, '{0}_test'.format(opt.net)))
	final_loss = test()
	torch.save(final_loss, os.path.join(opt.load_path, '{0}_test'.format(opt.net), 'loss.pt'))
else:
	if opt.load_path is not None:
		if opt.save_path is None:
			opt.save_path = opt.load_path
		vis_code = torch.load(os.path.join(opt.load_path, 'samples', 'vis_code.pt')).cuda()

		load_state(opt.load_path, 'last')
		index_shuffle = state['index_shuffle']
		current_iter = state['current_iter']
		best_iter = state['best_iter']
		min_loss = state['min_loss']
		current_sample = state['current_sample']
	else:
		if opt.save_path is None:
			raise ValeError('must specify save path if not continue training')
		if not os.path.exists(opt.save_path):
			os.makedirs(opt.save_path)
		for sub_folder in ('samples', 'running_test', 'net_archive', 'log'):
			if not os.path.exists(os.path.join(opt.save_path, sub_folder)):
				os.mkdir(os.path.join(opt.save_path, sub_folder))
		vis_code = torch.randn(opt.vis_row * opt.vis_col, opt.code_size).cuda()
		torch.save(vis_code, os.path.join(opt.save_path, 'samples', 'vis_code.pt'))

		index_shuffle = torch.randperm(train_index.size(0))
		current_iter = 0
		best_iter = 0
		min_loss = 1e100
		current_sample = 0

		vis_target = torch.Tensor(min(test_index.size(0), opt.vis_row * opt.vis_col), 3, opt.height, opt.width)
		for i in range(vis_target.size(0)):
			vis_target[i].copy_(get_data(test_index[i]))
		if opt.output_scale:
			torchvision.utils.save_image(vis_target * 2 - 1, os.path.join(opt.save_path, 'running_test', 'target.jpg'), opt.vis_row)
		else:
			torchvision.utils.save_image(vis_target, os.path.join(opt.save_path, 'running_test', 'target.jpg'), opt.vis_row)

	ones = Variable(torch.ones(opt.batch_size, 1).cuda())
	zeros = Variable(torch.zeros(opt.batch_size, 1).cuda())

	loss_record = torch.zeros(opt.test_interval, 3)

	visualize(vis_code, os.path.join(opt.save_path, 'samples', 'sample_{0}.jpg'.format(current_iter)))

	while current_iter < opt.niter:
		current_iter = current_iter + 1
		print('Iteration {0}:'.format(current_iter))
		current_loss_record = loss_record[(current_iter - 1) % opt.test_interval]

		for param in dis.parameters():
			param.requires_grad = True
		dis.zero_grad()

		true_sample = torch.Tensor(opt.batch_size, 3, opt.height, opt.width)
		for i in range(opt.batch_size):
			true_sample[i].copy_(get_data(train_index[index_shuffle[current_sample]]))
			current_sample = current_sample + 1
			if current_sample == train_index.size(0):
				current_sample = 0
				index_shuffle = torch.randperm(train_index.size(0))
		true_sample = Variable(true_sample.cuda())
		loss = lossfunc(dis(true_sample), ones)
		current_loss_record[0] = loss.data[0]
		loss.backward()
		del true_sample

		rand_code = Variable(torch.randn(opt.batch_size, opt.code_size).cuda())
		generated = gen(rand_code)
		loss = lossfunc(dis(generated), zeros)
		current_loss_record[1] = loss.data[0]
		generated.detach()
		loss.backward()

		dis_opt.step()

		for param in dis.parameters():
			param.requires_grad = False
		gen.zero_grad()

		rand_code = Variable(torch.randn(opt.batch_size, opt.code_size).cuda())
		loss = lossfunc(dis(gen(rand_code)), ones)
		current_loss_record[2] = loss.data[0]
		loss.backward()
		del generated

		gen_opt.step()

		print('loss: dis-real:{0} dis-fake:{1} gen:{2}'.format(current_loss_record[0], current_loss_record[1], current_loss_record[2]))

		if current_iter % opt.vis_interval == 0:
			visualize(vis_code, os.path.join(opt.save_path, 'samples', 'sample_{0}.jpg'.format(current_iter)))

		if current_iter % opt.test_interval == 0:
			print('Testing ...')
			current_loss = test()
			log = {
				'training_loss' : loss_record,
				'test_loss' : current_loss
			}
			torch.save(log, os.path.join(opt.save_path, 'log', 'loss_{0}.pt'.format(current_iter)))
			if current_loss < min_loss:
				print('new best network!')
				min_loss = current_loss
				best_iter = current_iter
				save_state(opt.save_path, 'best')
			save_state(opt.save_path, 'last')

		if current_iter % opt.save_interval == 0:
			save_state(opt.save_path, current_iter)
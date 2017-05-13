import argparse
import torch
import torchvision.datasets as datasets
import os
import os.path

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',                           required = True,
	help = 'cifar10 | lsun | imagenet | folder | lfw')

parser.add_argument('--lsun_class',                        default = 'bedroom',
	help = 'class of lsun dataset to use')

parser.add_argument('--dataroot',                          required = True,
	help = 'path to dataset')

parser.add_argument('--running',             type = int,   default = 200,
	help = 'number of samples to use in running test')

parser.add_argument('--final',               type = int,   default = 2000,
	help = 'number of samples to use in final test')

opt = parser.parse_args()

if opt.dataset == 'cifar10':
	dataset1 = datasets.CIFAR10(root = opt.dataroot, download = True)
	dataset2 = datasets.CIFAR10(root = opt.dataroot, train = False)
	fullsize = len(dataset1) + len(dataset2)
else:
	if opt.dataset in ['imagenet', 'folder', 'lfw']:
		dataset = datasets.ImageFolder(root = opt.dataroot)
	elif opt.dataset == 'lsun':
		dataset = datasets.LSUN(db_path = opt.dataroot, classes = [opt.lsun_class + '_train'])
	fullsize = len(dataset)

index_shuffle = torch.randperm(fullsize)

data_index = {}
data_index['running_test'] = index_shuffle[:opt.running].clone()
data_index['final_test'] = index_shuffle[:opt.final].clone()
data_index['train'] = index_shuffle[opt.final:].clone()
torch.save(data_index, os.path.join(opt.dataroot, 'data_index.pt'))

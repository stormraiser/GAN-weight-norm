import torch
import torch.nn as nn
from modules.WeightNormalizedConv import *
from modules.WeightNormalizedLinear import *
from modules.TPReLU import *
from modules.View import *

def build_discriminator(w_in, h_in, f_first, num_down_layers, norm):
	net = nn.Sequential()
	if (w_in % 2 != 0) or (h_in % 2 != 0):
		raise ValueError('input width and height must be even numbers')
	f_prev = 3
	f = f_first
	w = w_in
	h = h_in
	for i in range(num_down_layers):
		if i == num_down_layers - 1:
			pad_w = 0
			pad_h = 0
		else:
			if (w % 4 == 2):
				pad_w = 1
			else:
				pad_w = 0
			if (h % 4 == 2):
				pad_h = 1
			else:
				pad_h = 0
		if (norm == 'weight') or (norm == 'weight-affine'):
			net.add_module('level.{0}.conv'.format(i),
				WeightNormalizedConv2d(f_prev, f, 4, 2, (1 + pad_h, 1 + pad_w),
					scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
		else:
			net.add_module('level.{0}.conv'.format(i),
				nn.Conv2d(f_prev, f, 4, 2, (1 + pad_h, 1 + pad_w)))
		if (norm == 'batch') and (i > 0):
			net.add_module('level.{0}.batchnorm'.format(i),
				nn.BatchNorm2d(f))
		if norm == 'weight':
			net.add_module('level.{0}.tprelu'.format(i),
				TPReLU(f))
		else:
			net.add_module('level.{0}.prelu'.format(i),
				nn.PReLU(f))
		f_prev = f
		f = f * 2
		w = (w + pad_w * 2) // 2
		h = (h + pad_h * 2) // 2
	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('final.conv',
			WeightNormalizedConv2d(f_prev, 1, (h, w)))
	else:
		net.add_module('final.conv',
			nn.Conv2d(f_prev, 1, (h, w)))
	net.add_module('final.sigmoid', nn.Sigmoid())
	net.add_module('final.view', View(1))
	return net

def build_generator(w_out, h_out, f_last, num_up_layers, code_size, norm):
	net = nn.Sequential()
	if (w_out % 2 != 0) or (h_out % 2 != 0):
		raise ValueError('output width and height must be even numbers')
	pad_w = []
	pad_h = []
	w = w_out
	h = h_out
	f = f_last
	for i in range(num_up_layers - 1):
		if (w % 4 == 2):
			pad_w.append(1)
			w = (w + 2) // 2
		else:
			pad_w.append(0)
			w = w // 2
		if (h % 4 == 2):
			pad_h.append(1)
			h = (h + 2) // 2
		else:
			pad_h.append(0)
			h = h // 2
		f = f * 2
	w = w // 2
	h = h // 2
	pad_w.append(0)
	pad_h.append(0)

	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('initial.linear',
			WeightNormalizedLinear(code_size, f * h * w, init_factor = 0.01,
				scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
	else:
		net.add_module('initial.linear',
			nn.Linear(code_size, f * h * w))

	net.add_module('initial.view', View(f, h, w))

	if norm == 'batch':
		net.add_module('initial.batchnorm', nn.BatchNorm2d(f))
	if norm == 'weight':
		net.add_module('initial.tprelu', TPReLU(f))
	else:
		net.add_module('initial.prelu', nn.PReLU(f))

	for i in range(num_up_layers - 1):
		level = num_up_layers - 1 - i

		if (norm == 'weight') or (norm == 'weight-affine'):
			net.add_module('level.{0}.conv'.format(level),
				WeightNormalizedConvTranspose2d(f, f // 2, 4, 2, (1 + pad_h[level], 1 + pad_w[level]),
					scale = (norm == 'weight-affine'), bias = (norm == 'weight-affine')))
		else:
			net.add_module('level.{0}.conv'.format(level),
				nn.ConvTranspose2d(f, f // 2, 4, 2, (1 + pad_h[level], 1 + pad_w[level])))

		if norm == 'batch':
			net.add_module('level.{0}.batchnorm'.format(level),
				nn.BatchNorm2d(f // 2))
		if norm == 'weight':
			net.add_module('level.{0}.tprelu'.format(level),
				TPReLU(f // 2))
		else:
			net.add_module('level.{0}.prelu'.format(level),
				nn.PReLU(f // 2))

		f = f // 2

	if (norm == 'weight') or (norm == 'weight-affine'):
		net.add_module('level.0.conv',
			WeightNormalizedConvTranspose2d(f, 3, 4, 2, (1 + pad_h[0], 1 + pad_w[0])))
	else:
		net.add_module('level.0.conv',
			nn.ConvTranspose2d(f, 3, 4, 2, (1 + pad_h[0], 1 + pad_w[0])))
	net.add_module('level.0.sigmoid', nn.Sigmoid())
	return net
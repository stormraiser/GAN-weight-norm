import torch
from torch.nn.modules.module import Module

class View(Module):
	
	def __init__(self, *target_size):
		super(View, self).__init__()
		self.target_size = target_size

	def forward(self, input):
		return input.contiguous().view(input.size(0), *self.target_size)
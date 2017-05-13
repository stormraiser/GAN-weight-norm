import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

class _WeightNormalizedConvNd(_ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride,
			padding, dilation, transposed, output_padding, scale, bias, init_factor):
		super(_WeightNormalizedConvNd, self).__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, transposed, output_padding, 1, False)
		if scale:
			if transposed:
				self.scale = Parameter(torch.ones(1, self.out_channels, *((1,) * len(kernel_size))))
			else:
				self.scale = Parameter(torch.ones(self.out_channels, 1, *((1,) * len(kernel_size))))
		else:
			self.register_parameter('scale', None)
		if bias:
			if transposed:
				self.bias = Parameter(torch.zeros(1, self.out_channels, *((1,) * len(kernel_size))))
			else:
				self.bias = Parameter(torch.zeros(self.out_channels, 1, *((1,) * len(kernel_size))))
		else:
			self.register_parameter('bias', None)
		self.weight.data.mul_(init_factor)
		self.weight_norm_factor = 1.0
		if transposed:
			for t in self.stride:
				self.weight_norm_factor = self.weight_norm_factor / t

	def weight_norm(self):
		weight_norm = self.weight.pow(2)
		if self.transposed:
			weight_norm = weight_norm.sum(0)
		else:
			weight_norm = weight_norm.sum(1)
		for i in range(len(self.kernel_size)):
			weight_norm = weight_norm.sum(2 + i)
		weight_norm = weight_norm.mul(self.weight_norm_factor).add(1e-6).sqrt()
		return weight_norm

	def norm_scale_bias(self, input):
		if self.transposed:
			output = input.div(self.weight_norm().expand_as(input))
		else:
			output = input.div(self.weight_norm().transpose(0, 1).expand_as(input))
		if self.scale is not None:
			output = output.mul(self.scale.expand_as(input))
		if self.bias is not None:
			output = output.add(self.bias.expand_as(input))
		return output
			
	def __repr__(self):
		s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
			 ', stride={stride}')
		if self.padding != (0,) * len(self.padding):
			s += ', padding={padding}'
		if self.dilation != (1,) * len(self.dilation):
			s += ', dilation={dilation}'
		if self.output_padding != (0,) * len(self.output_padding):
			s += ', output_padding={output_padding}'
		if self.scale is None:
			s += ', scale=False'
		if self.bias is None:
			s += ', bias=False'
		s += ')'
		return s.format(name=self.__class__.__name__, **self.__dict__)

class WeightNormalizedConv2d(_WeightNormalizedConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, scale=True, bias=True, init_factor=1):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(WeightNormalizedConv2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			False, _pair(0), scale, bias, init_factor)

	def forward(self, input):
		return self.norm_scale_bias(F.conv2d(input, self.weight, None, self.stride,
						self.padding, self.dilation, 1))

class WeightNormalizedConvTranspose2d(_ConvTransposeMixin, _WeightNormalizedConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, output_padding=0, scale=True, bias=True, dilation=1, init_factor=1):
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		output_padding = _pair(output_padding)
		super(WeightNormalizedConvTranspose2d, self).__init__(
			in_channels, out_channels, kernel_size, stride, padding, dilation,
			True, output_padding, scale, bias, init_factor)

	def forward(self, input, output_size=None):
		output_padding = self._output_padding(input, output_size)
		return self.norm_scale_bias(F.conv_transpose2d(input, self.weight, None, self.stride, self.padding,
			output_padding, 1, self.dilation))

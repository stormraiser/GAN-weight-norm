import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class TPReLU(Module):

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(TPReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = Parameter(torch.zeros(num_parameters))

    def forward(self, input):
    	bias_resize = self.bias.view(1, self.num_parameters, *((1,) * (input.dim() - 2))).expand_as(input)
        return F.prelu(input - bias_resize, self.weight.clamp(0, 1)) + bias_resize

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
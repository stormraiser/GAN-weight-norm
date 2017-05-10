import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module

class WeightNormalizedLinear(Module):

    def __init__(self, in_features, out_features, scale=True, bias=True, init_factor=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.ones(out_features, 1))
        else:
            self.register_parameter('scale', None)
        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def normalized_weight(self):
        weight_norm = self.weight.pow(2).sum(1).add(1e-6).sqrt()
        if self.scale is not None:
            weight_norm = weight_norm.div(self.scale)
        normalized_weight = self.weight.div(weight_norm.expand_as(self.weight))
        return normalized_weight

    def forward(self, input):
        if self.bias is None:
            return F.linear(input, self.normalized_weight())
        else:
            return F.linear(input, self.normalized_weight(), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
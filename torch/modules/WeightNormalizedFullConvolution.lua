local torch = require 'torch'
local nn = require 'nn'
local WNFC, parent = torch.class('nn.WeightNormalizedFullConvolution', 'nn.Module')

function WNFC:__init(nInput, nOutput, kw, kh, dw, dh, pw, ph, hasScale, hasBias)
	parent.__init(self)
	self.conv = nn.SpatialFullConvolution(nInput, nOutput, kw, kh, dw, dh, pw, ph):noBias()
	self.weightNormFactor = math.sqrt(dw * dh)
	if hasScale == nil then
		self.hasScale = true
	else
		self.hasScale = hasScale
	end
	if hasBias == nil then
		self.hasBias = true
	else
		self.hasBias = hasBias
	end
	self.nInput = nInput
	self.nOutput = nOutput
	if self.hasScale then
		self.scale = torch.ones(nOutput)
		self.gradScale = torch.zeros(nOutput)
	end
	if self.hasBias then
		self.bias = torch.zeros(nOutput)
		self.gradBias = torch.zeros(nOutput)
	end
	self.eps = 1e-6
	self:updateStats()
end

function WNFC:updateStats()
	self.weightNorm = torch.sqrt(torch.add(torch.pow(self.conv.weight, 2):sum(1):sum(3):sum(4), self.eps)):reshape(self.nOutput)
end

function WNFC:updateOutput(input)
	self.conv:updateOutput(input)
	self.output = torch.cdiv(self.conv.output, self.weightNorm:reshape(1, self.nOutput, 1, 1):expandAs(self.conv.output))
	if self.hasScale then
		self.output:cmul(self.scale:reshape(1, self.nOutput, 1, 1):expandAs(self.conv.output))
	end
	self.output:mul(self.weightNormFactor)
	if self.hasBias then
		self.output:add(self.bias:reshape(1, self.nOutput, 1, 1):expandAs(self.conv.output))
	end
	return self.output
end

function WNFC:updateGradInput(input, gradOutput)
	self.gradTmp = gradOutput
	if self.hasScale then
		self.gradTmp:cmul(self.scale:reshape(1, self.nOutput, 1, 1):expandAs(gradOutput))
	end
	self.gradTmp:mul(self.weightNormFactor)
	self.gradTmp:cdiv(self.weightNorm:reshape(1, self.nOutput, 1, 1):expandAs(gradOutput))
	self.conv:updateGradInput(input, self.gradTmp)
	self.gradInput = self.conv.gradInput
	return self.gradInput
end

function WNFC:accGradParameters(input, gradOutput, scale)
	self.conv:accGradParameters(input, self.gradTmp, scale)
	self.conv.gradWeight:add(-scale, torch.cmul(self.conv.weight, torch.cdiv(torch.cmul(self.conv.output, self.gradTmp):sum(1):sum(3):sum(4):reshape(self.nOutput), torch.pow(self.weightNorm, 2)):reshape(1, self.nOutput, 1, 1):expandAs(self.conv.weight)))
	if self.hasScale then
		self.gradScale:add(scale * self.weightNormFactor, torch.cmul(gradOutput, torch.cdiv(self.conv.output, self.weightNorm:reshape(1, self.nOutput, 1, 1):expandAs(self.conv.output))):sum(1):sum(3):sum(4):reshape(self.nOutput))
	end
	if self.hasBias then
		self.gradBias:add(scale, gradOutput:sum(1):sum(3):sum(4):reshape(self.nOutput))
	end
end

function WNFC:zeroGradParameters()
	self.conv:zeroGradParameters()
	if self.hasScale then
		self.gradScale:zero()
	end
	if self.hasBias then
		self.gradBias:zero()
	end
end

function WNFC:updateParameters(learningRate)
	self.conv:updateParameters(learningRate)
	if self.hasScale then
		self.scale:add(-learningRate, self.gradScale)
	end
	if self.hasBias then
		self.bias:add(-learningRate, self.gradBias)
	end
end

function WNFC:type(__type)
	self.conv:type(__type)
	if self.hasScale then
		self.scale = self.scale:type(__type)
		self.gradScale = self.gradScale:type(__type)
	end
	if self.hasBias then
		self.bias = self.bias:type(__type)
		self.gradBias = self.gradBias:type(__type)
	end
	self.weightNorm = self.weightNorm:type(__type)
end

function WNFC:parameters()
	local param = {self.conv.weight}
	local grad = {self.conv.gradWeight}
	if self.hasScale then
		table.insert(param, self.scale)
		table.insert(grad, self.gradScale)
	end
	if self.hasBias then
		table.insert(param, self.bias)
		table.insert(grad, self.gradBias)
	end
	return param, grad
end

function WNFC:clearState()
	self.output = nil
	self.gradInput = nil
	self.gradTmp = nil
	self.conv:clearState()
end
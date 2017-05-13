local TPReLU, parent = torch.class('nn.TPReLU', 'nn.Module')

function TPReLU:__init(nFeatures, shareWeight, fixedWeight)
	parent.__init(self)
	self.nFeatures = nFeatures
	self.shareWeight = shareWeight
	self.fixedWeight = fixedWeight
	if shareWeight then
		self.weight = torch.ones(1) * 0.25
		self.gradWeight = torch.zeros(1)
	else
		self.weight = torch.ones(nFeatures) * 0.25
		self.gradWeight = torch.zeros(nFeatures)
	end
	if fixedWeight ~= nil then
		self.weight:fill(fixedWeight)
	end
	self.bias = torch.zeros(nFeatures)
	self.gradBias = torch.zeros(nFeatures)
end

function TPReLU:updateOutput(input)
	self.biasSize = input:size()
	self.weightSize = input:size()
	self.biasSize[1] = 1
	self.weightSize[1] = 1
	for i = 3, input:dim() do
		self.biasSize[i] = 1
		self.weightSize[i] = 1
	end
	if self.shareWeight then
		self.weightSize[2] = 1
	end
	self.output = input - torch.cmul(torch.cmax(self.bias:reshape(self.biasSize):expandAs(input) - input, 0), torch.csub(self.weight, 1):reshape(self.weightSize):expandAs(input))
	return self.output
end

function TPReLU:updateGradInput(input, gradOutput)
	local inputSignNeg = torch.cmax(torch.sign(self.bias:reshape(self.biasSize):expandAs(input) - input), 0)
	self.gradInput = torch.cmul(gradOutput, torch.cmul(self.weight:reshape(self.weightSize):expandAs(input), inputSignNeg) - torch.csub(inputSignNeg, 1))
	return self.gradInput
end

function TPReLU:accGradParameters(input, gradOutput, scale)
	local inputSignNegGradOutput = torch.cmul(torch.cmax(torch.sign(self.bias:reshape(self.biasSize):expandAs(input) - input), 0), gradOutput)
	if self.fixedWeight == nil then
		local tmp = torch.cmul(inputSignNegGradOutput, input - self.bias:reshape(self.biasSize):expandAs(input))
		for i = 3, input:dim() do
			tmp = tmp:sum(i)
		end
		tmp = tmp:sum(1):reshape(self.nFeatures)
		if self.shareWeight then
			tmp = tmp:sum(1)
		end
		self.gradWeight:add(scale, tmp)
	end
	local tmp = torch.cmul(inputSignNegGradOutput, -torch.csub(self.weight, 1):reshape(self.weightSize):expandAs(input))
	for i = 3, input:dim() do
		tmp = tmp:sum(i)
	end
	tmp = tmp:sum(1):reshape(self.nFeatures)
	self.gradBias:add(scale, tmp)
end

function TPReLU:zeroGradParameters()
	self.gradWeight:zero()
	self.gradBias:zero()
end

function TPReLU:type(__type)
	self.weight = self.weight:type(__type)
	self.bias = self.bias:type(__type)
	self.gradWeight = self.gradWeight:type(__type)
	self.gradBias = self.gradBias:type(__type)
end

function TPReLU:parameters()
	return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
end

function TPReLU:clearState()
	self.output = nil
	self.gradInput = nil
	self.inputSize = nil
end

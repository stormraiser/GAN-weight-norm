local WNL, parent = torch.class('nn.WeightNormalizedLinear', 'nn.Module')

function WNL:__init(nInput, nOutput, hasScale, hasBias, initWeightFactor)
	parent.__init(self)
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
	self.weight = torch.Tensor(nInput, nOutput)
	stdv = initWeightFactor / math.sqrt(nInput)
	self.weight:apply(function()
		return torch.uniform(-stdv, stdv)
	end)
	self.gradWeight = torch.zeros(nInput, nOutput)
	if self.hasScale then
		self.scale = torch.ones(nOutput)
		self.gradScale = torch.zeros(nOutput)
	end
	if self.hasBias then
		self.bias = torch.zeros(nOutput)
		self.gradBias = torch.zeros(nOutput)
	end
	self.eps = 1e-8
	self:updateStats()
end

function WNL:updateStats()
	self.weightNorm = torch.sqrt(torch.add(torch.pow(self.weight, 2):sum(1), self.eps))
end

function WNL:updateOutput(input)
	self.tmp = torch.cdiv(torch.mm(input, self.weight), self.weightNorm:expand(input:size(1), self.nOutput))
	self.output = self.tmp
	if self.hasScale then
		self.output:cmul(self.scale:reshape(1, self.nOutput):expand(input:size(1), self.nOutput))
	end
	if self.hasBias then
		self.output:add(self.bias:reshape(1, self.nOutput):expand(input:size(1), self.nOutput))
	end
	return self.output
end

function WNL:updateGradInput(input, gradOutput)
	self.gradTmp = gradOutput
	if self.hasScale then
		self.gradTmp:cmul(self.scale:reshape(1, self.nOutput):expand(input:size(1), self.nOutput))
	end
	self.gradInput = torch.mm(torch.cdiv(self.gradTmp, self.weightNorm:expand(input:size(1), self.nOutput)), self.weight:t())
	return self.gradInput
end

function WNL:accGradParameters(input, gradOutput, scale)
	self.gradWeight:add(
		scale,
		torch.cmul(
			torch.bmm(
				input:reshape(input:size(1), self.nInput, 1),
				torch.pow(self.weightNorm, -1):reshape(1, 1, self.nOutput):expand(input:size(1), 1, self.nOutput)
			)
			-
			torch.cmul(
				self.weight:reshape(1, self.nInput, self.nOutput):expand(input:size(1), self.nInput, self.nOutput),
				torch.cdiv(
					self.tmp,
					torch.pow(self.weightNorm, 2):reshape(1, self.nOutput):expand(input:size(1), self.nOutput)
				):reshape(input:size(1), 1, self.nOutput):expand(input:size(1), self.nInput, self.nOutput)
			),
			self.gradTmp:reshape(input:size(1), 1, self.nOutput):expand(input:size(1), self.nInput, self.nOutput)
		):sum(1)[1]
	)
	if self.hasScale then
		self.gradScale:add(scale, torch.cmul(self.tmp, gradOutput):sum(1)[1])
	end
	if self.hasBias then
		self.gradBias:add(scale, gradOutput:sum(1)[1])
	end
end

function WNL:zeroGradParameters()
	self.gradWeight:zero()
	if self.hasScale then
		self.gradScale:zero()
	end
	if self.hasBias then
		self.gradBias:zero()
	end
end

function WNL:updateParameters(learningRate)
	self.weight:add(-learningRate, self.gradWeight)
	if self.hasScale then
		self.scale:add(-learningRate, self.gradScale)
	end
	if self.hasBias then
		self.bias:add(-learningRate, self.gradBias)
	end
end

function WNL:type(__type)
	self.weight = self.weight:type(__type)
	self.gradWeight = self.gradWeight:type(__type)
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

function WNL:parameters()
	local param = {self.weight}
	local grad = {self.gradWeight}
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

function WNL:clearState()
	self.output = nil
	self.gradInput = nil
	self.tmp = nil
	self.gradTmp = nil
end
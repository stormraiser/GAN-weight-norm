local torch = require 'torch'
local nn = require 'nn'
require 'modules/WeightNormalizedLinear'
require 'modules/WeightNormalizedConvolution'
require 'modules/WeightNormalizedFullConvolution'
require 'modules/TPReLU'

function build_discriminator(w_in, h_in, f_first, num_down_layers, norm)
	net = nn.Sequential()
	f_prev = 3
	f = f_first
	w = w_in
	h = h_in
	for i = 1, num_down_layers do
		if i == num_down_layers then
			pad_w = 0
			pad_h = 0
		else
			if w % 4 == 2 then
				pad_w = 1
			else
				pad_w = 0
			end	
			if h % 4 == 2 then
				pad_h = 1
			else
				pad_h = 0
			end
		end
		if (norm == 'weight') or (norm == 'weight-affine') then
			net:add(nn.WeightNormalizedConvolution(f_prev, f, 4, 4, 2, 2, 1 + pad_w, 1 + pad_h, (norm == 'weight-affine'), (norm == 'weight-affine')))
		else
			net:add(nn.SpatialConvolution(f_prev, f, 4, 4, 2, 2, 1 + pad_w, 1 + pad_h))
		end
		if (norm == 'batch') and (i > 1) then
			net:add(nn.SpatialBatchNormalization(f))
		end
		if norm == 'weight' then
			net:add(nn.TPReLU(f))
		else
			net:add(nn.PReLU(f))
		end
		f_prev = f
		f = f * 2
		w = (w + pad_w * 2) / 2
		h = (h + pad_h * 2) / 2
	end
	if (norm == 'weight') or (norm == 'weight-affine') then
		net:add(nn.WeightNormalizedConvolution(f_prev, 1, w, h))
	else
		net:add(nn.SpatialConvolution(f_prev, 1, w, h))
	end
	net:add(nn.Sigmoid())
	net:add(nn.Reshape(1))
	return net
end

function build_generator(w_out, h_out, f_last, num_up_layers, code_size, norm)
	net = nn.Sequential()
	pad_w = {}
	pad_h = {}
	w = w_out
	h = h_out
	f = f_last
	for i = 1, num_up_layers - 1 do
		if (w % 4 == 2) then
			table.insert(pad_w, 1)
			w = (w + 2) / 2
		else
			table.insert(pad_w, 0)
			w = w / 2
		end
		if (h % 4 == 2) then
			table.insert(pad_h, 1)
			h = (h + 2) / 2
		else
			table.insert(pad_h, 0)
			h = h / 2
		end
		f = f * 2
	end
	w = w / 2
	h = h / 2
	table.insert(pad_w, 0)
	table.insert(pad_h, 0)

	if (norm == 'weight') or (norm == 'weight-affine') then
		net:add(nn.WeightNormalizedLinear(code_size, f * h * w, (norm == 'weight-affine'), (norm == 'weight-affine'), 0.01))
	else
		net:add(nn.Linear(code_size, f * h * w))
	end
	net:add(nn.Reshape(f, h, w))

	if norm == 'batch' then
		net:add(nn.SpatialBatchNormalization(f))
	end
	if norm == 'weight' then
		net:add(nn.TPReLU(f))
	else
		net:add(nn.PReLU(f))
	end

	for i = 1, num_up_layers - 1 do
		level = num_up_layers - i

		if (norm == 'weight') or (norm == 'weight-affine') then
			net:add(nn.WeightNormalizedFullConvolution(f, f / 2, 4, 4, 2, 2, 1 + pad_w[level], 1 + pad_h[level], (norm == 'weight-affine'), (norm == 'weight-affine')))
		else
			net:add(nn.SpatialFullConvolution(f, f / 2, 4, 4, 2, 2, 1 + pad_w[level], 1 + pad_h[level]))
		end

		if norm == 'batch' then
			net:add(nn.SpatialBatchNormalization(f / 2))
		end
		if norm == 'weight' then
			net:add(nn.TPReLU(f / 2))
		else
			net:add(nn.PReLU(f / 2))
		end

		f = f / 2
	end

	if (norm == 'weight') or (norm == 'weight-affine') then
		net:add(nn.WeightNormalizedFullConvolution(f, 3, 4, 4, 2, 2, 1 + pad_w[1], 1 + pad_h[1]))
	else
		net:add(nn.SpatialFullConvolution(f, 3, 4, 4, 2, 2, 1 + pad_w[1], 1 + pad_h[1]))
	end

	net:add(nn.Sigmoid())
	return net
end
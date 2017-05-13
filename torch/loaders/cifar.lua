local torch = require 'torch'
local paths = require 'paths'

function cifar_loader(dataroot)
	local train = torch.load(paths.concat(dataroot, 'cifar10-train.t7'))
	local test = torch.load(paths.concat(dataroot, 'cifar10-test.t7'))

	function get_data(index)
		if index <= train.data:size(1) then
			return train.data[index]:float() / 255
		else
			return test.data[index - train.data:size(1)]:float() / 255
		end
	end

	return get_data
end
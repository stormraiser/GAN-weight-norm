local torch = require 'torch'
local paths = require 'paths'
local image = require 'image'

function folder_loader(dataroot)
	local imageList = torch.load(paths.concat(dataroot, 'image_list.t7'))

	function get_data(index)
		return image.load(paths.concat(dataroot, imageList[index][1], imageList[index][2]))
	end

	return get_data
end
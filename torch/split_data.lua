local argparse = require 'argparse'
local torch = require 'torch'
local paths = require 'paths'
local lmdb = require 'lmdb'

parser = argparse()

parser:option('--dataset', 'cifar10 | lsun | folder')
parser:option('--lsun_class', 'class of lsun dataset to use', 'bedroom')
parser:option('--dataroot', 'path to dataset')
parser:option('--running', 'number of samples to use in running test', 200, tonumber)
parser:option('--final', 'number of samples to use in final test', 2000, tonumber)

opt = parser:parse()

if opt.dataset == 'cifar10' then
	local train = torch.load(paths.concat(opt.dataroot, 'cifar10-train.t7'))
	local test = torch.load(paths.concat(opt.dataroot, 'cifar10-test.t7'))
	fullsize = train.data:size(1) + test.data:size(1)
elseif opt.dataset == 'lsun' then
	local db = lmdb.env {
		Path = paths.concat(opt.dataroot, string.format('%s_train_lmdb', opt.lsun_class)),
		RDONLY = true
	}
	db:open()
	fullsize = db:stat().entries
else
	local imageList = {}
	for dir in paths.iterdirs(opt.dataroot) do
		for imageFile in paths.iterfiles(paths.concat(opt.dataroot, dir)) do
			table.insert(imageList, {dir, imageFile})
		end
	end
	fullsize = #imageList
	torch.save(paths.concat(opt.dataroot, 'image_list.t7'), imageList)
end
index_shuffle = torch.randperm(fullsize)

data_index = {}
data_index.running_test = index_shuffle:narrow(1, 1, opt.running)
data_index.final_test = index_shuffle:narrow(1, 1, opt.final)
data_index.train = index_shuffle:narrow(1, opt.final + 1, fullsize - opt.final)
torch.save(paths.concat(opt.dataroot, 'data_index.t7'), data_index)

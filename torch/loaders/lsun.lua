local torch = require 'torch'
local paths = require 'paths'
local image = require 'image'
local lmdb = require 'lmdb'
local tds = require 'tds'
local ffi = require 'ffi'

function lsun_loader(dataroot, class)
    local indexfile = paths.concat(dataroot, string.format('%s_hashes_chartensor.t7', class))

	local db = lmdb.env {
		Path = paths.concat(dataroot, string.format('%s_train_lmdb', class)),
		RDONLY = true
	}
	db:open()
	local reader = db:txn(true)

    if not paths.filep(indexfile) then
	    cursor = reader:cursor()
	    hashes = tds.hash()

	    count = 1
	    while true do
	       local key, data = cursor:get()
	       hashes[count] = key
	       count = count + 1
	       if not cursor:next() then
	           break
	       end
	    end

	    hashTensor = torch.CharTensor(#hashes, #hashes[1])
	    for i = 1, #hashes do
	    	ffi.copy(hsh2[i]:data(), hashes[i], #hashes[1])
	    end

	    torch.save(indexfile, hashTensor)
	else
		hashTensor = torch.load(indexfile)
	end

	function get_data(index)
		local hashString = ''
		for i = 1, 40 do
			hashString = hashString .. string.char(hashTensor[index][i])
		end
		local blob = reader:get(hashString)
		return image.decompress(blob, 3, 'float')
	end

	return get_data
end
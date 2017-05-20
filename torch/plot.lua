local argparse = require 'argparse'
local torch = require 'torch'
local gnuplot = require 'gnuplot'
local paths = require 'paths'

parser = argparse()
parser:option('--load_paths', 'paths to experiments to plot'):args('+')
parser:option('--type', 'test | dis | dis-real | dis-fake | gen', 'test')
parser:option('--interval', 'window size for averaged discriminator loss', 100, tonumber)
opt = parser:parse()
print(opt)

function comp(a, b)
	return a[1] < b[1]
end

plots = {}

for i, load_path in pairs(opt.load_paths) do
	logs = {}
	for log_file in paths.iterfiles(paths.concat(load_path, 'log')) do
		niter = tonumber(string.sub(log_file, 6, #log_file - 3))
		log = torch.load(paths.concat(load_path, 'log', log_file))
		if opt.type == 'test' then
			loss = log.test_loss
			table.insert(logs, {niter, loss})
		else
			losses = log.training_loss
			for i = 1, losses:size(1) / opt.interval do
				avg_loss = losses:narrow(1, (i - 1) * opt.interval + 1, opt.interval):mean(1)[1]
				if opt.type == 'dis' then
					loss = avg_loss[1] + avg_loss[2]
				elseif opt.type == 'dis-real' then
					loss = avg_loss[1]
				elseif opt.type == 'dis-fake' then
					loss = avg_loss[2]
				else
					loss = avg_loss[3]
				end
				table.insert(logs, {niter - losses:size(1) + i * opt.interval, loss})
			end
		end
	end
	table.sort(logs, comp)
	n = #logs
	x = torch.Tensor(n)
	y = torch.Tensor(n)
	for i = 1, n do
		x[i] = logs[i][1]
		y[i] = logs[i][2]
	end
	table.insert(plots, {paths.basename(load_path), x, y, 'with line'})
end

gnuplot.plot(plots)
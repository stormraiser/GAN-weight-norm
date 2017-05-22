local argparse = require 'argparse'
local torch = require 'torch'
local nn = require 'nn'
local optim = require 'optim'
local image = require 'image'
require 'cutorch'
require 'cunn'

require 'model'

local parser = argparse()

parser:option('--dataset', 'cifar10 | lsun | folder')
parser:option('--lsun_class', 'class of lsun dataset to use', 'bedroom')
parser:option('--dataroot', 'path to dataset')
parser:option('--batch_size', 'input batch size', 32, tonumber)
parser:option('--image_size', 'image size', -1, tonumber)
parser:option('--width', 'image width', -1, tonumber)
parser:option('--height', 'image height', -1, tonumber)
parser:option('--crop_size', 'crop size before scaling', -1, tonumber)
parser:option('--crop_width', 'crop width before scaling', -1, tonumber)
parser:option('--crop_height', 'crop height before scaling', -1, tonumber)
parser:option('--code_size', 'size of latent code', 128, tonumber)
parser:option('--nfeature', 'number of features of first conv layer', 64, tonumber)
parser:option('--nlayer', 'number of down/up conv layers', -1, tonumber)
parser:option('--norm', 'type of normalization: none | batch | weight | weight-affine', 'none')
parser:option('--save_path', 'path to save generated files')
parser:option('--load_path', 'load to continue existing experiment or evaluate')
parser:option('--lr', 'learning rate', 0.0001, tonumber)
parser:option('--test_interval', 'how often to test reconstruction', 1000, tonumber)
parser:option('--test_lr', 'learning rate for reconstruction test', 0.01, tonumber)
parser:option('--test_steps', 'number of steps in running reconstruction test', 50, tonumber)
parser:option('--vis_interval', 'how often to save generated samples', 100, tonumber)
parser:option('--vis_size', 'size of visualization grid', 10, tonumber)
parser:option('--vis_row', 'height of visualization grid', -1, tonumber)
parser:option('--vis_col', 'width of visualization grid', -1, tonumber)
parser:option('--save_interval', 'how often to save network', 20000, tonumber)
parser:option('--niter', 'number of iterations to train', 1000000, tonumber)
parser:flag('--final_test', 'do final test')
parser:option('--net', 'network to load for final test: best | last | <niter>', 'best')
parser:option('--gpu', 'id of the gpu to use', 1, tonumber)

function update_stats(module)
    if torch.typename(module) == 'nn.WeightNormalizedLinear'
        or torch.typename(module) == 'nn.WeightNormalizedConvolution'
        or torch.typename(module) == 'nn.WeightNormalizedFullConvolution' then
        module:updateStats()
    end
end

function prelu_clamp(module)
    if torch.typename(module) == 'nn.PReLU' or torch.typename(module) == 'nn.TPReLU' then
        module.weight:clamp(0, 1)
    end
end

local opt = parser:parse()
print(opt)
cutorch.setDevice(opt.gpu)

opt.tr_width = opt.width
opt.tr_height = opt.height
if opt.image_size > 0 then
    opt.width = opt.image_size
    opt.height = opt.image_size
end

local function transform(input)
    output = input
    if (opt.crop_height > 0) and (opt.crop_width > 0) then
        output = image.crop(output, "c", opt.crop_width, opt.crop_height)
    end
    if (opt.tr_width > 0) and (opt.tr_height > 0) then
        output = image.scale(output, string.format("%dx%d", opt.tr_width, opt.tr_height))
    else
        output = image.scale(output, string.format("^%d", opt.image_size))
        output = image.crop(output, "c", opt.image_size, opt.image_size)
    end
    if output:size(1) == 1 then
        output = output:expand(3, output:size(2), output:size(3))
    end
    return output
end

if (opt.vis_row <= 0) or (opt.vis_col <= 0) then
    opt.vis_row = opt.vis_size
    opt.vis_col = opt.vis_size
end

if opt.nlayer < 0 then
    opt.nlayer = 0
    s = math.max(opt.width, opt.height)
    while s >= 8 do
        s = math.floor((s + 1) / 2)
        opt.nlayer = opt.nlayer + 1
    end
end

if opt.dataset == 'cifar10' then
    require 'loaders/cifar'
    get_raw_data = cifar_loader(opt.dataroot)
elseif opt.dataset == 'lsun' then
    require 'loaders/lsun'
    get_raw_data = lsun_loader(opt.dataroot, opt.lsun_class)
else
    require 'loaders/folder'
    get_raw_data = folder_loader(opt.dataroot)
end

function get_data(index)
    return transform(get_raw_data(index))
end

data_index = torch.load(paths.concat(opt.dataroot, 'data_index.t7'))
train_index = data_index.train

if opt.final_test then
    test_index = data_index.final_test
else
    test_index = data_index.running_test
end

test_cri = nn.MSECriterion()
test_cri:cuda()

if not opt.final_test then
    train_cri = nn.BCECriterion()
    train_cri:cuda()
end

function init()
    gen = build_generator(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.code_size, opt.norm):cuda()
    gen:apply(update_stats)
    gen_state = {}
    dis = build_discriminator(opt.width, opt.height, opt.nfeature, opt.nlayer, opt.norm):cuda()
    dis:apply(update_stats)
    dis_state = {}
    state = {
        index_shuffle = torch.randperm(train_index:size(1)),
        current_iter = 0,
        best_iter = 0,
        min_loss = 1e100,
        current_sample = 0
    }

    vis_code = torch.randn(opt.vis_row * opt.vis_col, opt.code_size):cuda()
    torch.save(paths.concat(opt.save_path, 'samples', 'vis_code.t7'), vis_code)
end

function load(prefix, gen_only)
    gen = torch.load(paths.concat(opt.load_path, 'net_archive', prefix .. '_gen.t7')):cuda()

    if not gen_only then
        gen_state = torch.load(paths.concat(opt.load_path, 'net_archive', prefix .. '_gen_state.t7'))
        dis = torch.load(paths.concat(opt.load_path, 'net_archive', prefix .. '_dis.t7')):cuda()
        dis_state = torch.load(paths.concat(opt.load_path, 'net_archive', prefix .. '_dis_state.t7'))
        state = torch.load(paths.concat(opt.load_path, 'net_archive', prefix .. '_state.t7'))
        vis_code = torch.load(paths.concat(opt.load_path, 'samples', 'vis_code.t7')):cuda()
    end
end

function save(prefix)
    torch.save(paths.concat(opt.save_path, 'net_archive', prefix .. '_gen.t7'), gen:clearState())
    torch.save(paths.concat(opt.save_path, 'net_archive', prefix .. '_gen_state.t7'), gen_state)
    torch.save(paths.concat(opt.save_path, 'net_archive', prefix .. '_dis.t7'), dis:clearState())
    torch.save(paths.concat(opt.save_path, 'net_archive', prefix .. '_dis_state.t7'), dis_state)
    torch.save(paths.concat(opt.save_path, 'net_archive', prefix .. '_state.t7'), state)
end

function save_image(samples, filename)
    img = torch.Tensor(3, (opt.height + 2) * opt.vis_row, (opt.width + 2) * opt.vis_col)
    for i = 1, opt.vis_row do
        for j = 1, opt.vis_col do
            img:narrow(2, (i - 1) * (opt.height + 2) + 2, opt.height):narrow(3, (j - 1) * (opt.width + 2) + 2, opt.width):copy(samples[(i - 1) * opt.vis_col + j])
        end
    end
    image.save(filename, img)
end

function visualize(code, filename)
    gen:evaluate()
    generated = torch.Tensor(code:size(1), 3, opt.height, opt.width)
    for i = 1, math.floor((code:size(1) - 1) / opt.batch_size) + 1 do
        batch_size = math.min(opt.batch_size, code:size(1) - (i - 1) * opt.batch_size)
        batch_code = code:narrow(1, (i - 1) * opt.batch_size + 1, batch_size)
        generated:narrow(1, (i - 1) * opt.batch_size + 1, batch_size):copy(gen:forward(batch_code))
    end
    save_image(generated, filename)
    gen:training()
end

function test()
    local test_loss = 0
    local total_batch = math.floor((test_index:size(1) - 1) / opt.batch_size) + 1
    local best_code = torch.Tensor(test_index:size(1), opt.code_size):cuda()
    gen:evaluate()

    for i = 1, total_batch do
        if opt.final_test then
            print(string.format('Testing batch %d of %d ...', i, total_batch))
        end
        local batch_size = math.min(opt.batch_size, test_index:size(1) - (i - 1) * opt.batch_size)
        local batch_code_flat = torch.zeros(batch_size * opt.code_size):cuda()
        local batch_target = torch.Tensor(batch_size, 3, opt.height, opt.width):cuda()
        for j = 1, batch_size do
            batch_target[j]:copy(get_data(test_index[(i - 1) * opt.batch_size + j]))
        end

        local function batch_opt_func(code)
            local batch_code = code:reshape(batch_size, opt.code_size)
            local generated = gen:forward(batch_code)
            local batch_loss = test_cri:forward(generated, batch_target)
            local batch_grad = gen:backward(batch_code, test_cri:backward(generated, batch_target))
            return batch_loss, batch_grad:reshape(code:size(1))
        end
        local test_config = {
            learningRate = opt.test_lr,
            alpha = 0.9,
            epsilon = 1e-6
        }
        local test_state = {}
        for j = 1, opt.test_steps do
            batch_code_flat = optim.rmsprop(batch_opt_func, batch_code_flat, test_config, test_state)
        end
        best_code:narrow(1, (i - 1) * opt.batch_size + 1, batch_size):copy(batch_code_flat:reshape(batch_size, opt.code_size))
        
        local last_generated = gen:forward(batch_code_flat:reshape(batch_size, opt.code_size))
        local loss = test_cri:forward(last_generated, batch_target)
        test_loss = test_loss + loss * batch_size
        if opt.final_test then
            print(string.format('batch loss = %f', loss))
            local sample_rec_pair = torch.Tensor(3, opt.height + 2, (opt.width + 2) * 2)
            for j = 1, batch_size do
                sample_rec_pair:narrow(2, 2, opt.height):narrow(3, 2, opt.width):copy(batch_target[j])
                sample_rec_pair:narrow(2, 2, opt.height):narrow(3, opt.width + 4, opt.width):copy(last_generated[j])
                image.save(paths.concat(opt.load_path, opt.net .. '_test', string.format('%d.png', (i - 1) * opt.batch_size + j)), sample_rec_pair)
            end
        end
    end

    if not opt.final_test then
        visualize(best_code, paths.concat(opt.save_path, 'running_test', string.format('test_%d.jpg', state.current_iter)))
    end
    test_loss = test_loss / test_index:size(1)
    print(string.format('loss = %f', test_loss))
    gen:training()
    return test_loss
end

if opt.final_test then
    load(opt.net, true)
    if not paths.dirp(paths.concat(opt.load_path, opt.net .. '_test')) then
        paths.mkdir(paths.concat(opt.load_path, opt.net .. '_test'))
    end
    local final_loss = test()
    torch.save(final_loss, paths.concat(opt.load_path, opt.net .. '_test', 'loss.t7'))
else
    if opt.load_path ~= nil then
        if opt.save_path == nil then
            opt.save_path = opt.load_path
        end
        load('last', false)
    else
        if not paths.dirp(opt.save_path) then
            paths.mkdir(opt.save_path)
        end
        for k, sub_folder in pairs({'samples', 'running_test', 'net_archive', 'log'}) do
            if not paths.dirp(paths.concat(opt.save_path, sub_folder)) then
                paths.mkdir(paths.concat(opt.save_path, sub_folder))
            end
        end
        init()

        local vis_target = torch.Tensor(opt.vis_row * opt.vis_col, 3, opt.height, opt.width)
        for i = 1, opt.vis_row * opt.vis_col do
            vis_target[i]:copy(get_data(test_index[i]))
        end
        save_image(vis_target, paths.concat(opt.save_path, 'running_test', 'target.jpg'))
    end

    local dis_param, dis_grad = dis:getParameters()
    local gen_param, gen_grad = gen:getParameters()

    local ones = torch.ones(opt.batch_size, 1):cuda()
    local zeros = torch.zeros(opt.batch_size, 1):cuda()

    local loss_record = torch.zeros(opt.test_interval, 3)

    visualize(vis_code, paths.concat(opt.save_path, 'samples', string.format('sample_%d.jpg', state.current_iter)))

    train_config = {
        learningRate = opt.lr,
        alpha = 0.9,
        epsilon = 1e-6
    }

    while state.current_iter < opt.niter do
        state.current_iter = state.current_iter + 1
        print(string.format('Iteration %d:', state.current_iter))
        local current_loss_record = loss_record[(state.current_iter - 1) % opt.test_interval + 1]

        local function dis_opt_func(param)
            dis:zeroGradParameters()

            local sample = torch.Tensor(opt.batch_size, 3, opt.height, opt.width):cuda()
            for i = 1, opt.batch_size do
                state.current_sample = state.current_sample + 1
                sample[i]:copy(get_data(train_index[state.index_shuffle[state.current_sample]]))
                if state.current_sample == train_index:size(1) then
                    state.current_sample = 0
                end
            end
            local dis_output = dis:forward(sample)
            local dis_loss = train_cri:forward(dis_output, ones)
            dis:backward(sample, train_cri:backward(dis_output, ones))
            current_loss_record[1] = dis_loss

            local generated = gen:forward(torch.randn(opt.batch_size, opt.code_size):cuda())
            dis_output = dis:forward(generated)
            dis_loss = train_cri:forward(dis_output, zeros)
            dis:backward(generated, train_cri:backward(dis_output, zeros))
            current_loss_record[2] = dis_loss

            return current_loss_record[1] + current_loss_record[2], dis_grad
        end

        local function gen_opt_func(param)
            gen:zeroGradParameters()

            local code = torch.randn(opt.batch_size, opt.code_size):cuda()
            local generated = gen:forward(code)
            local dis_output = dis:forward(generated)
            local dis_loss = train_cri:forward(dis_output, ones)
            gen:backward(code, dis:backward(generated, train_cri:backward(dis_output, ones)))
            current_loss_record[3] = dis_loss

            return current_loss_record[3], gen_grad
        end

        dis_param:copy(optim.rmsprop(dis_opt_func, dis_param, train_config, dis_state))
        dis:apply(update_stats)
        dis:apply(prelu_clamp)
        gen_param:copy(optim.rmsprop(gen_opt_func, gen_param, train_config, gen_state))
        gen:apply(update_stats)
        gen:apply(prelu_clamp)

        print(string.format('loss: dis-real:%f dis-fake:%f gen:%f', current_loss_record[1], current_loss_record[2], current_loss_record[3]))

        if state.current_iter % opt.vis_interval == 0 then
            visualize(vis_code, paths.concat(opt.save_path, 'samples', string.format('sample_%d.jpg', state.current_iter)))
        end

        if state.current_iter % opt.test_interval == 0 then
            print('Testing ...')
            local current_loss = test()
            local log = {
                training_loss = loss_record,
                test_loss = current_loss
            }
            torch.save(paths.concat(opt.save_path, 'log', string.format('loss_%d.t7', state.current_iter)), log)
            if current_loss < state.min_loss then
                print('new best network!')
                state.min_loss = current_loss
                state.best_iter = state.current_iter
                save('best')
            end
            save('last')
        end

        if state.current_iter % opt.save_interval == 0 then
            save(tostring(state.current_iter))
        end
    end
end
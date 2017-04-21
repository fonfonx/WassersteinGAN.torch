--[[
    This file is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/main.lua).
]]--

require 'torch'
require 'nn'
require 'optim'

require 'model'

opt = {
    batchSize = 64,
    beta1 = 0.5,            -- momentum term of adam
    c = 0.01,               -- bound for weight clipping of the critic
    dataset = 'folder',     -- folder
    display = 6050,         -- port for displaying images during training / 0 = false
    display_id = 10,        -- display window id.
    gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    imgSize = 64,
    loadSize = 64,
    lr = 5e-5,              -- initial learning rate for adam
    name = 'wasserstein-gan',
    nc = 3,                 -- # number of channels of the input and generated images
    ndf = 64,               -- #  of discrim filters in first conv layer
    ngf = 64,               -- #  of gen filters in first conv layer
    ncritic = 5,            -- #  of training iterations of D for 1 iteration of G
    niter = 25,             -- #  of iter at starting learning rate
    noise = 'normal',       -- uniform / normal
    nThreads = 0,           -- #  of data loading threads to use
    ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
    nz = 100,               -- #  of dim for Z
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if opt.display then
    disp = require 'display'
    disp.configure({ hostname = '0.0.0.0', port = opt.display })
end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local nc = opt.nc
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = -1

-- load models
local netG = get_netG(nz, ngf, nc)
local netD = get_netD(nc, ndf)
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, opt.nc, opt.imgSize, opt.imgSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
    input = input:cuda()
    noise = noise:cuda()
    label = label:cuda()
    netD:cuda()
    netG:cuda()
    cudnn.benchmark = true
    cudnn.convert(netG, cudnn)
    cudnn.convert(netD, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- clamp parameters
   parametersD:clamp(-opt.c, opt.c)

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   errD_real = netD:forward(input)
   errD_real = errD_real:mean()
   netD:backward(input, label)

   -- train with fake
   if opt.noise == 'uniform' then
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   errD_fake = netD:forward(input)
   errD_fake = errD_fake:mean()
   netD:backward(input, label)

   errD = errD_real - errD_fake
   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   if opt.noise == 'uniform' then
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(real_label) -- fake labels are real for generator cost

   errG = netD:forward(input)
   errG = errG:mean()
   local df_dg = netD:updateGradInput(input, label)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

-- train
local counter = 0
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local i = 1
   local len_dataloader = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
   while i <= len_dataloader do
      tm:reset()
      -- train the critic longer at the beginning
      local Diter
      if counter <= 25 then
          Diter = 100
      else
          Diter = opt.ncritic
      end
      -- (1) Update critic
      for j = 1, Diter do
          optim.rmsprop(fDx, parametersD, optimStateD)
      end

      i = i + Diter

      -- (2) Update generator
      optim.rmsprop(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, { win = opt.display_id, title = 'Generated Images ' .. opt.name })
          disp.image(real, { win = opt.display_id + 1, title = 'Real Images ' .. opt.name })
      end

      -- logging
     print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
               .. '  Err_D: %.4f  Err_G: %.4f  Err_D_real: %.4f  Err_D_fake: %.4f'):format(
             epoch, i - 1, len_dataloader,
             tm:time().real, data_tm:time().real,
             errD, errG, errD_real, errD_fake))
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

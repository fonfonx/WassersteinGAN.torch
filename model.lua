--[[
    This file is adapted from the code of dcgan.torch software
    (see https://github.com/soumith/dcgan.torch/blob/master/main.lua).
]]--

local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.02)
        m:noBias()
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

function get_netG(nz, ngf, nc)
    local netG = nn.Sequential()
    -- input is Z, going into a convolution
    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    -- state size: (ngf*8) x 4 x 4
    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 8 x 8
    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 16 x 16
    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
    -- state size: (ngf) x 32 x 32
    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
    netG:add(nn.Tanh())
    -- state size: (nc) x 64 x 64

    netG:apply(weights_init)

    return netG
end

function get_netD(nc, ndf)
    local netD = nn.Sequential()

    -- input is (nc) x 64 x 64
    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
    netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 16 x 16
    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 8 x 8
    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
    netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
    -- netD:add(nn.Sigmoid())
    -- state size: 1 x 1 x 1
    netD:add(nn.View(1):setNumInputDims(3))
    -- state size: 1

    netD:apply(weights_init)

    return netD
end

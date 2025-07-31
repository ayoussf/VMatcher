import torch
import torch.nn as nn

class VGG_Block(nn.Module):
    def __init__(self, input_dim, output_dim, kn_size=3, pad=1, stride=2, 
                 batch_norm=True, activation=True, **factory_kwargs):
        super(VGG_Block, self).__init__()
        self.in_channels = input_dim
        self.layer = nn.Sequential()
        conv2d = nn.Conv2d(input_dim, output_dim, kernel_size=kn_size, stride=stride, padding=pad, **factory_kwargs)
        self.layer.add_module('conv', conv2d)
        
        if batch_norm:
            norm = nn.BatchNorm2d(output_dim, **factory_kwargs)
            self.layer.add_module('bn', norm)
        
        if activation:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.layer(x))

class VGG_BACKBONE(nn.Module):
    def __init__(self, config, device="cuda", dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(VGG_BACKBONE, self).__init__()

        dims = config.block_dims
        
        self.block_1 = VGG_Block(1, dims[0], stride=2, **factory_kwargs)
        self.block_2 = VGG_Block(dims[0], dims[0], stride=1, **factory_kwargs)
        self.block_3 = VGG_Block(dims[0], dims[0], stride=1, **factory_kwargs)

        self.block_4 = VGG_Block(dims[0], dims[1], stride=2, **factory_kwargs)
        self.block_5 = VGG_Block(dims[1], dims[1], stride=1, **factory_kwargs)
        self.block_6 = VGG_Block(dims[1], dims[1], stride=1, **factory_kwargs)

        self.block_7 = VGG_Block(dims[1], dims[2], stride=2, **factory_kwargs)
        self.block_8 = VGG_Block(dims[2], dims[2], stride=1, **factory_kwargs)
        self.block_9 = VGG_Block(dims[2], dims[2], stride=1, **factory_kwargs)
    
    def forward(self, x):
        x0 = self.block_1(x)
        x0 = self.block_2(x0)
        x0 = self.block_3(x0)

        x1 = self.block_4(x0)
        x1 = self.block_5(x1)
        x1 = self.block_6(x1)

        x2 = self.block_7(x1)
        x2 = self.block_8(x2)
        x2 = self.block_9(x2)
        return {'feats_c': x2, 'feats_f': None, 'feats_x2': x1, 'feats_x1': x0}

    def fuse_model(self):
        for m in self.modules():
            if type(m) == VGG_Block:
                m.layer = torch.nn.utils.fusion.fuse_conv_bn_eval(m.layer.conv, m.layer.bn)
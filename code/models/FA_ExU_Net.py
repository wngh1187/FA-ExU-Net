import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.l_num_convblocks = levels
    

    def forward(self, x, children=None):
        children = [] if children is None else children
        bottom = x
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x)
        if self.l_num_convblocks == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class Model(nn.Module):
    """
    RawNeXt model.
    This model is based on the deep layer aggregation (DLA) structure[1].

    Reference:
    [1] Yu, Fisher, et al. 
    "Deep layer aggregation." CVPR. 2018.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.l_channel = args['l_channel'] #[1, 1, 1, 2, 2, 1]
        self.l_num_convblocks = args['l_num_convblocks'] #[16, 16, 32, 64, 64, 128]
        self.code_dim = args['code_dim']
        self.stride = args['stride']
        self.first_kernel_size = args['first_kernel_size']
        self.first_stride_size = args['first_stride_size']
        self.first_padding_size = args['first_padding_size']
        
        if args['block'] == 'basic': 
            self.block=SEBasicBlock
        elif args['block'] == 'bottle': 
            self.block=SEBottleneck
      
        self.residual_root=False

        self.instancenorm   = nn.InstanceNorm1d(args['nfilts'])

        self.enc_base_layer = nn.Sequential(
            nn.Conv2d(1, self.l_channel[0], kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size, bias=False),
            nn.BatchNorm2d(self.l_channel[0]),
            nn.ReLU(inplace=True))

        self.n_level = len(self.l_channel)

        self.enc_level0 = self._make_conv_level(
            self.l_channel[0], self.l_channel[0], self.l_num_convblocks[0], self.stride[0])
        self.enc_level1 = self._make_conv_level(
            self.l_channel[0], self.l_channel[1], self.l_num_convblocks[1], self.stride[1])

        self.enc_level2 = Tree(self.l_num_convblocks[2], self.block, self.l_channel[1], self.l_channel[2], self.stride[2],
                           level_root=False, root_residual=self.residual_root)
        self.enc_level3 = Tree(self.l_num_convblocks[3], self.block, self.l_channel[2], self.l_channel[3], self.stride[3],
                           level_root=True, root_residual=self.residual_root)
        self.enc_level4 = Tree(self.l_num_convblocks[4], self.block, self.l_channel[3], self.l_channel[4], self.stride[4],
                           level_root=True, root_residual=self.residual_root)
        self.enc_level5 = Tree(self.l_num_convblocks[5], self.block, self.l_channel[4], self.l_channel[5], self.stride[5],
                           level_root=True, root_residual=self.residual_root)

        self.dec_level0 = Tree(self.l_num_convblocks[5], self.block, self.l_channel[5], self.l_channel[5], self.stride[5],
                            level_root=True, root_residual=self.residual_root)
        self.dec_conv0 = nn.ConvTranspose2d(self.l_channel[5]*2, self.l_channel[4], kernel_size=2, stride=2, bias=False)
        self.dec_level1 = Tree(self.l_num_convblocks[4], self.block, self.l_channel[4], self.l_channel[4], self.stride[4],
                           level_root=True, root_residual=self.residual_root)
        self.dec_conv1 = nn.Conv2d(self.l_channel[4]*2, self.l_channel[3], kernel_size=1, bias=False)
        self.dec_level2 = Tree(self.l_num_convblocks[3], self.block, self.l_channel[3], self.l_channel[3], self.stride[3],
                           level_root=True, root_residual=self.residual_root)
        self.dec_conv2 = nn.ConvTranspose2d(self.l_channel[3]*2, self.l_channel[2], kernel_size=2, stride=2, bias=False)
        self.dec_level3 = Tree(self.l_num_convblocks[2], self.block, self.l_channel[2], self.l_channel[2], self.stride[2],
                           level_root=False, root_residual=self.residual_root)
        self.dec_conv3 = nn.Conv2d(self.l_channel[2]*2, self.l_channel[1], kernel_size=1, bias=False)
        self.dec_level4 = self._make_conv_level(
            self.l_channel[1], self.l_channel[1], self.l_num_convblocks[1], self.stride[1])
        self.dec_conv4 = nn.ConvTranspose2d(self.l_channel[1]*2, self.l_channel[0], kernel_size=2, stride=2, bias=False)
        self.dec_level5 = self._make_conv_level(
            self.l_channel[0], self.l_channel[0], self.l_num_convblocks[0], self.stride[0])
        self.dec_conv5 = nn.Conv2d(self.l_channel[0]*2, self.l_channel[0], kernel_size=1, bias=False)

        self.dec_base_layer = nn.ConvTranspose2d(self.l_channel[0]*2, 1, kernel_size=(2,1), stride = (2,1), bias=False)
    

        self.ext_base_layer = nn.Sequential(
            nn.Conv2d(1, self.l_channel[0], kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size, bias=False),
            nn.BatchNorm2d(self.l_channel[0]),
            nn.ReLU(inplace=True))

        self.ext_level0 = self._make_conv_level(
            self.l_channel[0]*3, self.l_channel[0], self.l_num_convblocks[0], self.stride[0])
        self.ext_level1 = self._make_conv_level(
            self.l_channel[0]*3, self.l_channel[1], self.l_num_convblocks[1], self.stride[1])

        self.ext_level2 = Tree(self.l_num_convblocks[2], self.block, self.l_channel[1]*3, self.l_channel[2], self.stride[2],
                           level_root=False, root_residual=self.residual_root)
        self.ext_level3 = Tree(self.l_num_convblocks[3], self.block, self.l_channel[2]*3, self.l_channel[3], self.stride[3],
                           level_root=True, root_residual=self.residual_root)
        self.ext_level4 = Tree(self.l_num_convblocks[4], self.block, self.l_channel[3]*3, self.l_channel[4], self.stride[4],
                           level_root=True, root_residual=self.residual_root)
        self.ext_level5 = Tree(self.l_num_convblocks[5], self.block, self.l_channel[4]*3, self.l_channel[5], self.stride[5],
                           level_root=True, root_residual=self.residual_root)


        self.msfa_conv1 = nn.Sequential(
            nn.Conv2d(self.l_channel[5], self.l_channel[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.l_channel[2]),
            nn.ReLU(inplace=False))
        self.msfa_conv2 = nn.Sequential(
            nn.Conv2d(self.l_channel[4], self.l_channel[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.l_channel[2]),
            nn.ReLU(inplace=False))
        self.msfa_conv3 = nn.Sequential(
            nn.Conv2d(self.l_channel[3], self.l_channel[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.l_channel[2]),
            nn.ReLU(inplace=False))
        self.msfa_conv4 = nn.Sequential(
            nn.Conv2d(self.l_channel[2], self.l_channel[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.l_channel[2]),
            nn.ReLU(inplace=False))
        self.msfa_conv5 = nn.Sequential(
            nn.Conv2d(self.l_channel[1], self.l_channel[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.l_channel[2]),
            nn.ReLU(inplace=False))
        self.agg_conv = nn.Sequential(
            nn.Conv2d(self.l_channel[3]*5, self.l_channel[-1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(self.l_channel[-1]),
            nn.ReLU(inplace=False))


        final_dim = self.l_channel[-1] * 4    
        
        self.attention = nn.Sequential(
            nn.Conv1d(final_dim, final_dim//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_dim//8),
            nn.Conv1d(final_dim//8, final_dim, kernel_size=1), 
            nn.Softmax(dim=-1),
        )

        self.bn_agg= nn.BatchNorm1d(final_dim * 2)

        self.fc = nn.Linear(final_dim*2, self.code_dim)
        self.bn_code = nn.BatchNorm1d(self.code_dim)

        self.mp = nn.MaxPool2d(2)
        


    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
                ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, only_code = False):
        x = self.instancenorm(x).unsqueeze(1).detach()
        dic_encoder_x = {}
        dic_decoder_x = {}
        dic_extractor_x = {}
        
        x = self.enc_base_layer(x)
        base_x = x
        for i in range(self.n_level):
            x = getattr(self, 'enc_level{}'.format(i))(x)
            if i % 2 == 1: x = self.mp(x)
            dic_encoder_x['%d'%i] = x


        # +++++++++++++++++++ Expanding Path  +++++++++++++++++++++ #
        x = dic_encoder_x['%d'%(self.n_level-1)]
        for i in range(0, self.n_level):    
            x = getattr(self, 'dec_level{}'.format(i))((x))
            x = torch.cat((x, dic_encoder_x['%d'%(self.n_level-1-i)]), 1)
            x = getattr(self, 'dec_conv{}'.format(i))((x))
            dic_decoder_x['%d'%i] = x
            
        
        x = torch.cat((x, base_x), 1)     
        x = self.dec_base_layer(x)
        output = x
        x = self.ext_base_layer(x)

        for i in range(0, self.n_level):
            enc_x = base_x if i == 0 else dic_encoder_x['%d'%(i-1)]
            x = torch.cat((x, dic_decoder_x['%d'%(self.n_level-i-1)], enc_x), 1)            
            x = getattr(self, 'ext_level{}'.format(i))(x)
            if i % 2 == 1: x = self.mp(x)
            dic_extractor_x['%d'%i] = x
        
        ######
        # Feature pyramid module
        #####
        ex5 = dic_extractor_x['5']
        ex5 = self.msfa_conv1(ex5)

        ex4 = dic_extractor_x['4']
        ex4 = self.msfa_conv2(ex4)
        ex5_us = self._upsample(ex5, ex4)
        ex4 += ex5_us

        ex3 = dic_extractor_x['3']
        ex3 = self.msfa_conv3(ex3)
        ex3 += ex4

        ex2 = dic_extractor_x['2']
        ex2 = self.msfa_conv4(ex2)
        ex3_us = self._upsample(ex3, ex2)
        ex2 += ex3_us

        ex1 = dic_extractor_x['1']
        ex1 = self.msfa_conv5(ex1)
        ex1 += ex2

        
        #####
        # MSFA
        #####
        ex2_ds = self.mp(ex2)
        ex1_ds = self.mp(ex1)
        x = torch.cat((ex1_ds, ex2_ds, ex3, ex4, ex5_us), 1)
        x = self.agg_conv(x)

        
        bs, _, _, time = x.size()
        x = x.reshape(bs, -1, time)
            
        w = self.attention(x)
        m = torch.sum(x * w, dim=-1)
        s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
        x = torch.cat([m, s], dim=1)
        x =    self.bn_agg(x)
        
        code = self.fc(x)
        
        code = self.bn_code(code)

        if only_code:    return code
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 7.0
        code = torch.div(code, code_norm)
        return code, output

    def _upsample(self, x, y):      
        '''Upsample and add two feature maps.
        Args:
            x: (Variable) top feature map to be upsampled.
            y: (Variable) lateral feature map.
        Returns:
            (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,f,t = y.size()        
        
        x = F.upsample(x, size=(f,t), mode='bilinear')

        return x
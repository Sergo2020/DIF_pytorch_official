from torch import nn
import torch.nn.functional as F
import torch


class Conv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride,
                 padding=0, dilation=1, bias=True, activ=None, norm=None,
                 pool=None):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size=kernel,
                                               stride=stride, dilation=dilation, padding=padding, bias=bias))

        if activ == 'leak':
            activ = nn.LeakyReLU(inplace=True)
        elif activ == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activ == 'pleak':
            activ = nn.PReLU()
        elif activ == 'gelu':
            activ = nn.GELU()
        elif activ == 'selu':
            activ = nn.SELU()
        elif activ == 'sigmoid':
            activ = nn.Sigmoid()
        elif activ == 'softmax':
            activ = nn.Softmax(dim=1)
        elif activ == 'tanh':
            activ = nn.Tanh()
        if norm == 'bn':
            norm = nn.BatchNorm2d(out_c)
        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)

        if not norm is None:
            self.conv.add_module('norm', norm)

        if not pool is None:
            self.conv.add_module('pool', pool)

        if not activ is None:
            self.conv.add_module('activ', activ)

    def forward(self, x):
        x = self.conv(x)
        return x


class DeConv_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride,
                 padding=0, activ=None, norm=None,
                 pool=None, bias=True):
        super(DeConv_Layer, self).__init__()
        self.deconv = nn.Sequential()
        self.deconv.add_module('deconv', nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel,
                                                            stride=stride, padding=padding, bias=bias))

        if activ == 'leak':
            activ = nn.LeakyReLU(inplace=True)
        elif activ == 'relu':
            activ = nn.ReLU(inplace=True)
        elif activ == 'pleak':
            activ = nn.PReLU()
        elif activ == 'gelu':
            activ = nn.GELU()
        elif activ == 'selu':
            activ = nn.SELU()
        elif activ == 'sigmoid':
            activ = nn.Sigmoid()
        elif activ == 'softmax':
            activ = nn.Softmax(dim=1)
        if norm == 'bn':
            norm = nn.BatchNorm2d(out_c)
        if pool == 'max':
            pool = nn.MaxPool2d(2, 2)
        elif pool == 'avg':
            pool = nn.AvgPool2d(2, 2)

        if not norm is None:
            self.deconv.add_module('norm', norm)

        if not pool is None:
            self.deconv.add_module('pool', pool)

        if not activ is None:
            self.deconv.add_module('activ', activ)

    def forward(self, x):
        x = self.deconv(x)
        return x


class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, activ=None, pool=None, norm='bn'):
        super(Conv_Block, self).__init__()
        self.c1 = Conv_Layer(in_c, out_c, 3, 1, activ=activ, norm=norm, padding=1)
        self.c2 = Conv_Layer(out_c, out_c, 3, 1, activ=activ, norm=norm, padding=1)

        if pool == 'up_stride':
            self.pool = DeConv_Layer(out_c, out_c, 2, 2, norm=norm)
        elif pool == 'up_bilinear':
            self.pool = nn.Upsample(scale_factor=2, mode=pool[3:], align_corners=True)
        elif pool == 'up_nearest':
            self.pool = nn.Upsample(scale_factor=2, mode=pool[3:], align_corners=True)
        elif pool == 'down_max':
            self.pool = nn.MaxPool2d(2, 2)
        elif pool == 'down_stride':
            self.c2 = Conv_Layer(out_c, out_c, 3, 2, activ=activ, norm=norm, padding=1)
            self.pool = None
        else:
            self.pool = None

    def forward(self, x):
        x = self.c2(self.c1(x))

        if self.pool:
            return x, self.pool(x)
        else:
            return 0, x


# -------- Functions ----------------------------------------------------------

def concat_curr(prev, curr):
    diffY = prev.size()[2] - curr.size()[2]
    diffX = prev.size()[3] - curr.size()[3]

    curr = F.pad(curr, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

    x = torch.cat([prev, curr], dim=1)
    return x

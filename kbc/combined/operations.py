import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, emb_dim, affine, dropout=0: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, emb_dim, affine, dropout=0: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, emb_dim, affine, dropout=0: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, emb_dim, affine, dropout=0: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, emb_dim, affine, dropout=0: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, emb_dim, affine, dropout=0: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, emb_dim, affine, dropout=0: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, emb_dim, affine, dropout=0: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, emb_dim, affine, dropout=0: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_3x3'     : lambda C, stride, emb_dim, affine, dropout=0: Conv(C, C, 3, stride, 1, affine=affine),
  'conv_5x5'     : lambda C, stride, emb_dim, affine, dropout=0: Conv(C, C, 5, stride, 2, affine=affine),
  'conv_7x7'     : lambda C, stride, emb_dim, affine, dropout=0: Conv(C, C, 7, stride, 3, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, emb_dim, affine, dropout=0: nn.Sequential(
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine),
    nn.ReLU(inplace=False),
    nn.Dropout2d(p=0.2)
    ),
  'relu' : lambda C, stride, emb_dim, affine, dropout=0.3: ReLUOp(C, emb_dim, dropout),
  'tanh' : lambda C, stride, emb_dim, affine, dropout=0.3: tanhOp(C, emb_dim, dropout),
  'sigmoid': lambda C, stride, emb_dim, affine, dropout=0.3: sigmoidOp(C, emb_dim, dropout),
  'identity' : lambda C, stride, emb_dim, affine, dropout=0: Identity()
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=False),
      nn.Dropout2d(p=0.2)
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Dropout2d(p=0.2),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=False),
      nn.Dropout2d(p=0.2)
      )

  def forward(self, x):
    return self.op(x)


class Conv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(Conv, self).__init__()
    self.op = nn.Sequential(
      #nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=False),
      nn.Dropout2d(p=0.2)
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

# class Linear(nn.Module):

#   def __init__(self):
#     super(Linear, self).__init__()

#   def forward(self, x):
#     self.shape = x.shape
#     x = x.flatten()

class ReLUOp(nn.Module):

  def __init__(self, C, emb_dim, affine=True, dropout=0):
    super(ReLUOp, self).__init__()
    self.C = C
    self.op = nn.Sequential(
      nn.Linear(emb_dim,emb_dim),
      nn.ReLU(inplace=False),
      nn.Dropout(p=dropout)
      #nn.BatchNorm1d(emb_dim, affine=affine)
      )
    #self.bn = nn.BatchNorm2d(C, affine=affine)

  def forward(self, x):
    #x = x.contiguous().view([x.size(0),1,-1])
    x = self.op(x)
    #x = x.view([x.size(0),self.C,32,-1])
    #x = self.bn(x)

    return x

class tanhOp(nn.Module):

  def __init__(self, C, emb_dim, affine=True, dropout=0):
    super(tanhOp, self).__init__()
    self.C = C
    self.op = nn.Sequential(
      nn.Linear(emb_dim,emb_dim),
      nn.Tanh(),
      nn.Dropout(p=dropout)
      #nn.BatchNorm1d(emb_dim, affine=affine)
      )

  def forward(self, x):
    #x = x.contiguous().view([x.size(0),1,-1])
    x = self.op(x)
    #x = x.view([x.size(0),self.C,32,-1])
    #x = self.bn(x)

    return x

class sigmoidOp(nn.Module):

  def __init__(self, C, emb_dim, affine=True, dropout=0):
    super(sigmoidOp, self).__init__()
    self.C = C
    self.op = nn.Sequential(
      nn.Linear(emb_dim,emb_dim),
      nn.Sigmoid(),
      nn.Dropout(p=dropout)
      #nn.BatchNorm1d(emb_dim, affine=affine)
      )

  def forward(self, x):
    #x = x.contiguous().view([x.size(0),1,-1])
    x = self.op(x)
    #x = x.view([x.size(0),self.C,32,-1])
    #x = self.bn(x)

    return x


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


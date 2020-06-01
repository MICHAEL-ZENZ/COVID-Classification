import torch
import torch.nn as nn
import numpy as np

class prunnableConv2D(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size:int,
                stride:int=1,padding:int=0, dilation:int=1, groups=1,
                bias=True, padding_mode='zeros'):
    super(prunnableConv2D, self).__init__()

    kernel_size = (kernel_size,kernel_size)
    stride = (stride,stride)
    padding = (padding,padding)
    dilation = (dilation,dilation)

    self.in_channels=in_channels
    self.out_channels=out_channels
    self.kernel_size=kernel_size
    self.stride=stride
    self.padding=padding
    self.dilation=dilation
    self.groups=groups
    self.bias=bias
    self.padding_mode=padding_mode

    
    self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
      stride=stride,padding=padding, dilation=dilation, groups=groups,
      bias=bias, padding_mode=padding_mode)
    self.mask=np.ones(self.conv.weight.shape)
    
  def forward(self, x):

    weights=self.conv.weight.data.cpu().numpy()
    weights=weights*self.mask
    self.conv.weight.data=torch.from_numpy(weights.astype(np.float32)).cuda()
    x=self.conv(x)

    return x

  def setPruneRatio(self, ratio=0.0):
    if not (ratio>=0 and ratio <=1):
      raise ValueError("Invalid Prune Ratio:%f"%ratio)
    absweights=np.abs(self.conv.weight.data.cpu().numpy())
    weights_sort=absweights.reshape(-1)
    weights_sort.sort()
    thresh=weights_sort[int(len(weights_sort)*ratio)]
    self.mask[absweights<thresh]=0
    self.mask[absweights>=thresh]=1
  
  def resetPruneRatio(self):
    self.mask[True]=1

class prunnableLinear(nn.Module):
  def __init__(self, in_channels, out_channels, bias=True):
    super(prunnableLinear, self).__init__()

    self.in_channels=in_channels
    self.out_channels=out_channels
    self.bias=bias
    
    self.linear=nn.Linear(in_channels,out_channels,bias=bias)
    self.mask=np.ones(self.linear.weight.shape)
    
  def forward(self, x):

    weights=self.linear.weight.data.cpu().numpy()
    weights=weights*self.mask
    self.linear.weight.data=torch.from_numpy(weights.astype(np.float32)).cuda()
    x=self.linear(x)

    return x

  def setPruneRatio(self, ratio=0.0):
    if not (ratio>=0 and ratio <=1):
      raise ValueError("Invalid Prune Ratio:%f"%ratio)
    absweights=np.abs(self.linear.weight.data.cpu().numpy())
    weights_sort=absweights.reshape(-1)
    weights_sort.sort()
    thresh=weights_sort[int(len(weights_sort)*ratio)]
    self.mask[absweights<thresh]=0
    self.mask[absweights>=thresh]=1
  
  def resetPruneRatio(self):
    self.mask[True]=1
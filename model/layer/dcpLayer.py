
import torch
import torch.nn as nn

import numpy as np
from tensorly import decomposition as dcp
from tensorly import tenalg as tlg

def decompose(tensor:np.array, R1, R2, kernel_first=True):
  core,factors=None,None
  if kernel_first:
    core, factors = dcp.partial_tucker(tensor,(2,3),rank=(R1,R2),verbose=False)
    return (factors[0],core,factors[1])
  else:
    core, factors = dcp.partial_tucker(tensor,(0,1),rank=(R2,R1),verbose=False)
    return (factors[1],core,factors[0])

def reverseTrans(I,core,O):
	return tlg.multi_mode_dot(core,(I,O),(2,3),transpose=False)

class dcpConv2D(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size:int,
                R1_ratio=1.0, R2_ratio=1.0,
                stride:int=1,padding:int=0, dilation:int=1, groups=1,
                bias=True, padding_mode='zeros'):
    super(dcpConv2D, self).__init__()

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

    self.R1, self.R2=int(round(in_channels*R1_ratio)),int(round(out_channels*R2_ratio))
    self.conv1=nn.Conv2d(in_channels,self.R1,kernel_size=1,
      stride=1,padding=0,bias=False)
    self.conv2=nn.Conv2d(self.R1,self.R2,kernel_size=kernel_size,
      stride=stride,padding=padding,bias=False)
    self.conv3=nn.Conv2d(self.R2,out_channels,kernel_size=1,stride=1,
      padding=0,bias=bias)
    
  def forward(self, x):

    x=self.conv1(x)
    x=self.conv2(x)
    x=self.conv3(x)

    return x

  def loadStateFromModule(self, layer:nn.Conv2d):
    unmatchSettings=[]
    if not self.in_channels==layer.in_channels:
      unmatchSettings.append('in_channels')
    if not self.out_channels==layer.out_channels:
      unmatchSettings.append('out_channels')
    if not self.kernel_size==layer.kernel_size:
      unmatchSettings.append('kernel_size')
    if not self.stride==layer.stride:
      unmatchSettings.append('stride')
    if not self.padding==layer.padding:
      unmatchSettings.append('padding')
    if not self.dilation==layer.dilation:
      unmatchSettings.append('dilation')
    if not self.groups==layer.groups:
      unmatchSettings.append('groups')
    bias=False if layer.bias is None else True
    if not self.bias==bias:
      unmatchSettings.append('bias')
    if not self.padding_mode==layer.padding_mode:
      unmatchSettings.append('padding_mode')
    if len(unmatchSettings)!=0:
      raise ValueError('input layer does not share same setting on {}'.format(unmatchSettings))

    I,core,O=decompose(layer.weight.data.numpy(),self.R1,self.R2,kernel_first=False)
    I=I.transpose()
    I=I.reshape(I.shape+(1,1))
    O=O.reshape(O.shape+(1,1))
    self.conv1.weight.data=torch.from_numpy(I.copy())
    self.conv2.weight.data=torch.from_numpy(core.copy())
    self.conv3.weight.data=torch.from_numpy(O.copy())
    if self.bias:
      self.conv3.bias.data.copy_(layer.bias.data)

def tuckerDCP_Conv2D(in_channels, out_channels, kernel_size:int,
                R1_ratio=1.0, R2_ratio=1.0,
                stride:int=1,padding:int=0, dilation:int=1, groups=1,
                bias=True, padding_mode='zeros'):
  if kernel_size>1 and in_channels>=16 and out_channels>=16:
    return dcpConv2D(in_channels, out_channels, kernel_size,
                R1_ratio, R2_ratio,
                stride, padding, dilation, groups,
                bias, padding_mode)
  else:
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                stride,padding, dilation, groups,
                bias, padding_mode)
 #===================================================== Import libraries ================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn 
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ================================================= Conv2 Network ================================================================================
class conv2model(nn.Module):
  def __init__(self):
    super(conv2model,self).__init__()

    self.block1 = nn.Sequential(
                  nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 3,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3, padding =1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2))
    self.block2 = nn.Sequential(
                  nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 3,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size = 3, padding =1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2))



    self.fc =     nn.Sequential(
                  nn.Linear(8192,256),
                  nn.ReLU(),
                  nn.Linear(256,10), )
                  
                  


  def forward(self,x):
    out = self.block1(x)
    out = self.block2(out)
    out = out.view(out.size(0),-1)
    out = self.fc(out)

    return out
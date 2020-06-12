import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================= Flexible Layer Block1 ================================================================================
class FlexiLayer1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = kernel_size
        stride = stride
        padding = padding
        dilation = dilation
        super(FlexiLayer1, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        
        self.threshold1 = nn.parameter.Parameter(torch.randn((100, 64, 28, 28)), requires_grad=True)
            
    def forward(self, t):
        
        t_1 = F.relu(F.conv2d(t, self.weight, padding = self.padding)) # get convolution result
        t_2 = F.max_pool2d(t, kernel_size=self.kernel_size, stride=1, padding = self.padding) # get max result with the same kernel size
        #t_2 = torch.cat((t_2, t_2, t_2), 1)
        m = nn.Sigmoid()
        cond = torch.sub(t_2, self.threshold1)
        t_2 = m(cond*50)*t_2 # 
        t_1 = m(cond*(-50))*t_1 # 
        t = torch.add(t_2, t_1)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        return t
    
    # ================================================= Flexible Layer Block2 ================================================================================
class FlexiLayer2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = kernel_size
        stride = stride
        padding = padding
        dilation = dilation
        super(FlexiLayer2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        
        self.threshold1 = nn.parameter.Parameter(torch.randn((100, 128, 24, 24)), requires_grad=True)
            
    def forward(self, t):
        
        t_1 = F.relu(F.conv2d(t, self.weight, padding = self.padding)) # get convolution result
        t_2 = F.max_pool2d(t, kernel_size=self.kernel_size, stride=1, padding = self.padding) # get max result with the same kernel size
        #t_2 = torch.cat((t_2, t_2, t_2), 1)
        m = nn.Sigmoid()
        cond = torch.sub(t_2, self.threshold1)
        t_2 = m(cond*50)*t_2 # 
        t_1 = m(cond*(-50))*t_1 # 
        t = torch.add(t_2, t_1)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        return t
    
    # ================================================= Flexible Layer Block3 ================================================================================
class FlexiLayer3(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = kernel_size
        stride = stride
        padding = padding
        dilation = dilation
        super(FlexiLayer3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        
        self.threshold1 = nn.parameter.Parameter(torch.randn((100, 256, 20, 20)), requires_grad=True)
            
    def forward(self, t):
        
        t_1 = F.relu(F.conv2d(t, self.weight, padding = self.padding)) # get convolution result
        t_2 = F.max_pool2d(t, kernel_size=self.kernel_size, stride=1, padding = self.padding) # get max result with the same kernel size
        #t_2 = torch.cat((t_2, t_2, t_2), 1)
        m = nn.Sigmoid()
        cond = torch.sub(t_2, self.threshold1)
        t_2 = m(cond*50)*t_2 # 
        t_1 = m(cond*(-50))*t_1 # 
        t = torch.add(t_2, t_1)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        return t
    
    # ================================================= Flexible Layer Block4(2) ================================================================================

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class FlexiLayer4_2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = kernel_size
        stride = stride
        padding = padding
        dilation = dilation
        super(FlexiLayer4_2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        
        
        self.threshold1 = nn.parameter.Parameter(torch.randn((100, 512, 4, 4)), requires_grad=True)
            
    def forward(self, t):
        
        t_1 = F.relu(F.conv2d(t, self.weight, padding = self.padding)) # get convolution result
        t_2 = F.max_pool2d(t, kernel_size=self.kernel_size, stride=1, padding = self.padding) # get max result with the same kernel size
        #t_2 = torch.cat((t_2, t_2, t_2), 1)
        m = nn.Sigmoid()
        cond = torch.sub(t_2, self.threshold1)
        t_2 = m(cond*50)*t_2 # 
        t_1 = m(cond*(-50))*t_1 # 
        t = torch.add(t_2, t_1)
        #t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        return t
    
    # ================================================= VGG-16 Network ================================================================================
class VGG16(nn.Module):
  def __init__(self):
    super(VGG16,self).__init__()

    self.block1 = nn.Sequential(
                  nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  FlexiLayer1(in_channels = 64,out_channels = 64,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(64),
                  nn.ReLU(),
                  #nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.3))

    self.block2 = nn.Sequential(
                  nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(128),
                  nn.ReLU(),
                  FlexiLayer2(in_channels = 128,out_channels = 128,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(128),
                  nn.ReLU(),
                  #nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.4))

    self.block3 = nn.Sequential(
                  nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 256,out_channels = 256,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  FlexiLayer3(in_channels = 256,out_channels = 256,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.4))

    self.block4 = nn.Sequential(
                  nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,padding = 0),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  FlexiLayer4_2(in_channels = 512,out_channels = 512,kernel_size = 3, padding =0),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2) ,
                  nn.Dropout2d(0.4))

    self.block5 = nn.Sequential(
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3,padding = 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.Conv2d(in_channels = 512,out_channels = 512,kernel_size = 3, padding =1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout2d(0.5) )

    self.fc =     nn.Sequential(
                  nn.Linear(512,100),
                  nn.Dropout(0.5),
                  nn.BatchNorm1d(100),
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  nn.Linear(100,10), )
                  
                  


  def forward(self,x):
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out = self.block4(out)
    out = self.block5(out)
    out = out.view(out.size(0),-1)
    out = self.fc(out)

    return out

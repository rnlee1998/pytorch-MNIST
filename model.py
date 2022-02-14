import torch.nn as nn
import torch
class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
                                        nn.AvgPool2d(kernel_size=2,stride=2),
                                        nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
                                        nn.AvgPool2d(kernel_size=2,stride=2),
                                        nn.Flatten(),
                                        nn.Linear(16*25,120),nn.Sigmoid(),
                                        nn.Linear(120,84),nn.Sigmoid(),
                                        nn.Linear(84,10))
        
    
    def forward(self,x):
        return self.sequential(x)

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(nn.Flatten(),
                          nn.Linear(28*28,120),nn.Sigmoid(),
                          nn.Linear(120,84),nn.Sigmoid(),
                          nn.Linear(84,10))
        
    
    def forward(self,x):
        return self.sequential(x)

class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3,stride=2),
                                        nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3,stride=2),
                                        nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
                                        nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
                                        nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3,stride=2),
                                        nn.Flatten(),
                                        nn.Linear(6400,4096),nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096,4096),nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096,10))
        
    
    def forward(self,x):
        return self.sequential(x)

class VGG(nn.Module):
    def __init__(self,conv_arch) -> None:
        super().__init__()
        conv_blks=[]     
        in_channels=1
        for (num_convs,out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
            in_channels = out_channels
        self.sequential = nn.Sequential(*conv_blks,nn.Flatten(),
                             nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
                             nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
                             nn.Linear(4096,10))
           
    def forward(self,x):
        return self.sequential(x)
        
def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().init()
        self.conv1 = nn.Conv2d(input_channels,num_channels,
                               kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels,num_channels,
                               kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,
                                   kernel_size=1,stride = strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self,x):
        Y = nn.ReLU(self.bn1(self.conv1(x)))
        Y = nn.ReLU(self.bn2(self.conv2(Y)))
        if conv3:
            x = self.conv3(x)
        Y = Y + x
        return nn.ReLU(Y)

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    
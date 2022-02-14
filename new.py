import torch
from model import *
net = LeNet()
for name,param in net.named_parameters():
     print(name,' ',param.shape)
     break

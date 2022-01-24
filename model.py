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
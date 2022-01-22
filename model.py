import torch.nn as nn
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
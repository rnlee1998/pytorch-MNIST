import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
def MyDataloader(batch,mode='train',shuffle=False):   
    root = r"./"
    trans_compose  = transforms.Compose([transforms.ToTensor(),
                        ])
    if mode=='train':
        dataset = torchvision.datasets.MNIST(root,train=True,transform=trans_compose,download=True)
    else:
        dataset = torchvision.datasets.MNIST(root,train=False,transform=trans_compose,download=True)
    dataset_loader = DataLoader(dataset,batch_size=batch,shuffle=shuffle)
    return dataset_loader

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
def MyDataloader(batch,model,mode='train',shuffle=False):   
    root = r"./"
    if model=='AlexNet' or model=='VGG':
       trans_compose  = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),
                        ]) 
    else:
        trans_compose  = transforms.Compose([transforms.ToTensor(),
                        ])

    if mode=='train':
        dataset = torchvision.datasets.MNIST(root,train=True,transform=trans_compose,download=True)
    else:
        dataset = torchvision.datasets.MNIST(root,train=False,transform=trans_compose,download=True)
    dataset_loader = DataLoader(dataset,batch_size=batch,shuffle=shuffle)
    return dataset_loader

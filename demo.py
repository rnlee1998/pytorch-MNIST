import cv2
import matplotlib.pyplot as plt
from model import *
import torch
from PIL import Image
import numpy as np
import argparse
import os
import sys
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

parser = argparse.ArgumentParser(description='demo', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--model',default='ResNet',help='choose from LeNet, MLP,AlexNet,VGG')
parser.add_argument('--checkpoints',default='./checkpoints',help='model save dir')
if sys.argv.__len__() == 2:
  arg_filename_with_prefix = '@' + sys.argv[1]
  args = parser.parse_args([arg_filename_with_prefix])
else:
  args = parser.parse_args()

device = torch.device('cuda')
if args.model=='MLP':
    model = MLP()
    model.load_state_dict(torch.load("./checkpoints/MLP_best.pth"))
elif args.model=='LeNet':
    model = LeNet()
    model.load_state_dict(torch.load("./checkpoints/LeNet_best.pth"))
elif args.model=='AlexNet':
    model = AlexNet()
    model.load_state_dict(torch.load("./checkpoints/AlexNet_best.pth"))
elif args.model=='VGG':
    model = VGG([(1,64),(1,128),(2,256),(2,512),(2,512)]).to(device)
    model.load_state_dict(torch.load("./checkpoints/VGG_best.pth"))
elif args.model=='ResNet':
    model = ResNet()
    model.load_state_dict(torch.load("./checkpoints/ResNet_best.pth")) 
model.to(device)
model.eval()
if args.model=='MLP' or args.model=='LeNet':
    resize_H,resize_W=28,28
elif args.model=='AlexNet' or args.model=='VGG' or args.model=='ResNet':
    resize_H,resize_W=224,224

images_np = cv2.imread("./R-C.png",cv2.IMREAD_GRAYSCALE)
h,w = images_np.shape
images_np = np.array(255*torch.ones(h,w))-images_np#图片反色
images = Image.fromarray(images_np)
test_images = []
for i in range(10):
  for j in range(16):
    test_images.append(images_np[h//10*i:h//10+h//10*i,w//16*j:w//16*j+w//16])

correct = 0
i = 0
cnt = 1
output_list=[]
for sample in test_images:
  sample_tensor = torch.tensor(sample).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
  sample_tensor = torch.nn.functional.interpolate(sample_tensor,(resize_H,resize_W))
  predict = model(sample_tensor)
  output = predict.argmax()
  output_list.append(int(output))
  if(output==i):
    correct+=1
  if(cnt%16==0):
    i+=1
  cnt+=1
acc_g = correct/len(test_images)
print(f'acc_g:{acc_g}')
print(np.array(output_list).reshape(10,16))

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import tqdm as tqdm
import wandb
from model import *
from dataload import MyDataloader
import argparse
import sys
import os

def main():
  device = torch.device('cuda')
  batch = args.batch
  if args.model=='LeNet':
      model = LeNet()
  else:
      model = MLP()
  model.to(device)
  model.load_state_dict(torch.load("./checkpoints/LeNet_best.pth"))
  print(f"test the {args.model}...") 
  test_loader = MyDataloader(batch,mode = 'test',shuffle=False)
  with torch.no_grad():
        correct_test=0
        for x,y in test_loader:
            [x,y] = [x.to(device),y.to(device)]
            predict = model(x)
            correct_test += (predict.argmax(dim=1)==y).sum()
        acc_test = correct_test/(batch*len(test_loader))    
        print(f'acc_test:{acc_test}')

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='test argument set', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--batch',default=64, type=int,help='batch size')
    parser.add_argument('--model',default='LeNet',help='choose from LeNet, MLP')
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    main()
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
# wandb.init(project="pytorch-intro")# 项目名称

def main():
  device = torch.device('cuda')
  epochs = args.epochs
  batch = args.batch
  lr=0.9
  loss = nn.CrossEntropyLoss().to(device)
  if args.model=='LeNet':
      model = LeNet()
  else:
      model = MLP()
  model.to(device)
  print(f"train the {args.model}...")
  optimizer = torch.optim.SGD(model.parameters(),lr) 
  train_loader = MyDataloader(batch,mode = 'train',shuffle=True)
  test_loader = MyDataloader(batch,mode = 'test',shuffle=False)
  best_metric = float('-inf')
  for epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    correct_train = 0
    correct_test = 0
    for x,y in train_loader:
      x = x.to(device)
      y = y.to(device)
      predict = model(x)
      L = loss(predict,y)
      optimizer.zero_grad()
      L.backward()
      optimizer.step()
      train_loss = train_loss + L
      correct_train += (predict.argmax(dim=1)==y).sum()
    acc_train = correct_train/(batch*len(train_loader))
    with torch.no_grad():
      for x,y in test_loader:
        [x,y] = [x.to(device),y.to(device)]
        predict = model(x)
        L1 = loss(predict,y)
        test_loss = test_loss + L1
        correct_test += (predict.argmax(dim=1)==y).sum()
      acc_test = correct_test/(batch*len(test_loader))
      if acc_test>best_metric:
          is_best = True
          best_metric = acc_test
      else:
          is_best = False     
    # wandb.log({
    #       "Test Accuracy": acc_test,
    #       "Test Loss": test_loss
    #   })
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    if is_best:    
        torch.save(model.state_dict(),os.path.join(args.checkpoints,args.model+f'_best.pth'))
    print(f'epoch:{epoch},train_loss:{train_loss/batch},test_loss:{test_loss/batch},acc_train:{acc_train},acc_test:{acc_test}')


    

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='train argument set', fromfile_prefix_chars='@')
  parser.convert_arg_line_to_args = convert_arg_line_to_args
  parser.add_argument('--batch',default=64, type=int,help='batch size')
  parser.add_argument('--epochs',default=15, type=int,help='train epochs')
  parser.add_argument('--model',default='LeNet',help='choose from LeNet, MLP')
  parser.add_argument('--checkpoints',default='./checkpoints',help='model save dir')
  if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
  else:
    args = parser.parse_args()

  main()
import cv2
import matplotlib.pyplot as plt
from model import *
import torch
from PIL import Image
import numpy as np
device = torch.device('cuda')
model = LeNet().to(device)
model.load_state_dict(torch.load("./checkpoints/LeNet_best.pth"))
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
for sample in test_images:
  sample_tensor = torch.tensor(sample).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
  sample_tensor = torch.nn.functional.interpolate(sample_tensor,(28,28))
  predict = model(sample_tensor)
  output = predict.argmax()
  if(output==i):
    correct+=1
  if(cnt%16==0):
    i+=1
  cnt+=1
acc_g = correct/len(test_images)
print(f'acc_g:{acc_g}')

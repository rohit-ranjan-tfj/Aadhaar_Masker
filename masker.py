import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import jaccard_score
from pathlib import Path
import os
import re
import argparse
import pytesseract
from tqdm import tqdm
from test import *  

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--test_folder', default='./images/', type=str, help='folder path to input images')
parser.add_argument('--output_folder', default='./output/', type=str, help='folder path to output images')

args = parser.parse_args()

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class LeNet(Module):

    def __init__(self, numChannels=3):
        super(LeNet, self).__init__()
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
            kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
            kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = Linear(in_features=50*11*11 , out_features=512)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=512, out_features=20)
        self.relu4 = ReLU()
        self.fc3 = Linear(in_features=20, out_features=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x,1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)

        x = self.fc3(x)
        output = torch.sigmoid(x)  

        return output

def dist(a,b,c,d):
  return np.sqrt((a-c)**2 + (b-d)**2)

def replace_chars(text):
  """
  Replaces all characters instead of numbers from 'text'.
  
  :param text: Text string to be filtered
  :return: Resulting number
  """
  list_of_numbers = re.findall(r'\d+', text)
  result_number = ''.join(list_of_numbers)
  return result_number


if __name__ == '__main__':

  call_test(test_folder=args.test_folder,cuda=args.cuda)
  path_length=len(args.test_folder)
  if (args.test_folder[-1]!='/'):
    path_length=path_length+1
  paths=[]
  for dirname, _, filenames in os.walk(args.test_folder):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
  try:
    os.mkdir(args.output_folder)
  except:
    pass
  if(args.cuda==False):
    device='cpu'
  else:
    device='cuda'
  model = LeNet()
  model.load_state_dict(torch.load("./cnn_model",map_location=torch.device(device)))
  model = model.to(device)
  model.eval()
  pbar = tqdm(paths)
  for IMAGE_PATH in pbar:
    num_list = []
    image = cv2.imread(IMAGE_PATH)
    height = image.shape[0]
    width = image.shape[1]
    with open("./result/res_" + IMAGE_PATH[path_length:-4]+".txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            a,b,c,d,e,f,g,h=currentline
            a,b,c,d,e,f,g,h=int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h)

            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
            cv2.fillPoly(mask, points, (255))

            res = cv2.bitwise_and(image,image,mask = mask)

            rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
            crop_image = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            try:
              crop_image=cv2.resize(crop_image,(50,50))
            except:
              continue
            crop_image=crop_image.transpose(2,0,1)
            crop_image=crop_image.reshape(1,3,50,50)
            crop_image = torch.from_numpy(crop_image.astype(np.float32))
            crop_image = crop_image.to(device)            

            x=2.15
            y=2.65
 
            if (((dist(a,b,c,d)/dist(c,d,e,f)>=1/y) and (dist(a,b,c,d)/dist(c,d,e,f)<=1/x)) ):        
              num_list.append((a,b,c,d,e,f,g,h))
            elif (model(crop_image).item()>=0.5):
              num_list.append((a,b,c,d,e,f,g,h))
    for (a,b,c,d,e,f,g,h) in num_list:
      points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
      cv2.fillPoly(image, points, (0,0,0))
    cv2.imwrite(args.output_folder+"/masked_"+IMAGE_PATH[path_length:],image)

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
  if (args.output_folder[-1]!='/'):
    args.output_folder=args.output_folder+"/"    
  paths=[]
  for dirname, _, filenames in os.walk(args.test_folder):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
  try:
    os.mkdir(args.output_folder)
  except:
    pass
  
  pbar = tqdm(paths)
  for IMAGE_PATH in pbar:
    num_list = []
    image = cv2.imread(IMAGE_PATH)
    try:
      height = image.shape[0]
      width = image.shape[1]
    except:
      continue
    with open("./result/res_" + IMAGE_PATH[path_length:-4]+".txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            try:
              a,b,c,d,e,f,g,h=currentline
            except:
              continue
            a,b,c,d,e,f,g,h=int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h)

            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
            cv2.fillPoly(mask, points, (255))

            res = cv2.bitwise_and(image,image,mask = mask)

            rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
            crop_image = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
            try:
              crop_image= cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            except:
              continue
            string = pytesseract.image_to_string(crop_image,lang='eng',config='--psm 6')

            x=2.15
            y=2.65
 
            if (((dist(a,b,c,d)/dist(c,d,e,f)>=1/y) and (dist(a,b,c,d)/dist(c,d,e,f)<=1/x)) ):        
              num_list.append((a,b,c,d,e,f,g,h))
            elif (replace_chars(string)!= ""):
              string=string.strip()
              if (len(string)%4==0 and len(string)>=1):
                num_list.append((a,b,c,d,e,f,g,h))
    for (a,b,c,d,e,f,g,h) in num_list:
      points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
      cv2.fillPoly(image, points, (0,0,0))
    cv2.imwrite(args.output_folder+"masked_"+IMAGE_PATH[path_length:],image)
  print("Task completed successfully!")

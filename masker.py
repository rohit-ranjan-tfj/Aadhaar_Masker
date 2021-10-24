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
    area_list = []
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

            x=2.15
            y=2.65
            aspect=0.0

            if dist(a,b,c,d)<dist(c,d,e,f) :
              aspect=dist(a,b,c,d)/dist(c,d,e,f)
            else:
              aspect=dist(c,d,e,f)/dist(a,b,c,d)
              
            if (((aspect>=1/y) and (aspect<=1/x)) ):        
              num_list.append((a,b,c,d,e,f,g,h))
              calc = ( 0.5 * (a*d - c*b + c*f - e*d + e*h - g*f + g*b - a*h) )
              area_list.append(calc)
            else :
              mask = np.zeros((height, width), dtype=np.uint8)
              points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
              cv2.fillPoly(mask, points, (255))

              res = cv2.bitwise_and(image,image,mask = mask)

              rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
              crop_image = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
              string = pytesseract.image_to_string(crop_image,lang='eng',config='--oem 3 --psm 6')
              if (replace_chars(string)!= ""):
                string=string.strip()
                if (len(string)==4 or len(string)==8):
                  calc = ( 0.5 * (a*d - c*b + c*f - e*d + e*h - g*f + g*b - a*h) )
                  area_list.append(calc)
                  num_list.append((a,b,c,d,e,f,g,h))
                                  
    mean = np.mean(area_list)
    std = np.std(area_list)
    for i in range(len(num_list)):
      (a,b,c,d,e,f,g,h) = num_list[i]
      area = area_list[i]
      if len(num_list)>3 and (((area-mean)/std)>2 or ((area-mean)/std)<-1):
        continue
      points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
      cv2.fillPoly(image, points, (0,0,0))
    cv2.imwrite(args.output_folder+"masked_"+IMAGE_PATH[path_length:],image)
  print("Task completed successfully!")

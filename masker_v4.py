import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import jaccard_score
from pathlib import Path
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode
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

def dist(a,b,c,d): #Calculates the distance between (a,b) and (c,d)
    return np.sqrt((a-c)**2 + (b-d)**2)

def scale_image(img, scale_factor): #Scale factor is percent of original size e.g 0.3
    width = max(int(img.shape[1] * scale_factor),1)
    height = max(int(img.shape[0] * scale_factor),1)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def crop_image(img, points): #Crops the image to required pplygon
    
    #Creating a binary mask over the cropped area
    mask_img = np.full_like(img, 255, dtype=np.uint8)
    cv2.fillPoly(mask_img, points, (0))
    
    #Removing background outside of crop
    res = cv2.bitwise_or(img, mask_img)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    
    #Cropping Image
    crop = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    
    return crop

def detect_angle(image): #Deskews image using Hough Line Transform
    # convert to edges
    edges = cv2.Canny(image, 100, 200)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
        
    try:
        return float(skew_angle)
    except:
        return 0.0
    
def rotate_image(image, angle):
    
    if angle == 0.0:
        return image
    
    #Rotating image with new image dimensions to prevent accidental cropping out parts of img
    (h, w) = image.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - centre[0]
    M[1, 2] += (nH / 2) - centre[1]
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    return rotated

def tesseract_preprocess(img, pts, angle):
    
    #Best results if we scale via rectangular points for rotated images and actual image size for non-rotated
    if angle:
        #Finding height of image using coordinates of points
        v_distance = dist(pts[0][0][0], pts[0][0][1], pts[0][3][0], pts[0][3][1])
        h_distance = dist(pts[0][0][0], pts[0][0][1], pts[0][1][0], pts[0][1][1])
        height = min(v_distance, h_distance)

        scale_factor = 30.0/height #Scales the image to 33 pixels, ideal size for tesseract
    else:
        #Finding height via actual pixel size
        height = img.shape[0]
        scale_factor = 30.0/height
    
    scaled = scale_image(img, scale_factor)
            
    try:
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)[1]
        return gray, thresh
    except:
        return scaled, scaled #To avoid Occasional random cv2 error
    
def num_length(string):
    nums = re.sub('[^0-9?TBaAZ$egiG]','', string) #Removing non-numeric characters. However, include characters like ?  
    return len(nums)                              #that tesseract commonly interprets numbers as

def aspect_ratio(a,b,c,d,e,f,g,h):
    def dist(a,b,c,d):
        return np.sqrt((a-c)**2 + (b-d)**2)
    ar = dist(a,b,c,d)/dist(c,d,e,f) #Checking Aspect Ratio
    return ar

def ar_check(ar): #Checks if the crop falls in possible dimensions. Otherwise, we can skip that crop and speed up process
    if 1.65<=ar<=5.75: #Dimensions that we need to check
        return True
    else:
        return False
    
def mask_number(img, points): #Masks the image at the given points
    cv2.fillPoly(img, points, (0,0,0))
    return

def neighbouring_boxes(pts1, pts2, angle): #Checks if two boxes are next to each other and accounts for angle

    v_distance = dist(pts1[0][0][0], pts1[0][0][1], pts1[0][3][0], pts1[0][3][1])
    h_distance = dist(pts1[0][0][0], pts1[0][0][1], pts1[0][1][0], pts1[0][1][1])
    distance = dist(pts1[0][0][0], pts1[0][0][1], pts2[0][0][0], pts2[0][0][1])
    
    #The trigonometric terms account for angle
    if ((abs(pts1[0][0][1]-pts2[0][0][1])-abs(distance*math.sin(math.radians(angle))))<=0.75*v_distance) and ((abs(pts1[0][0][0]-pts2[0][0][0])+distance*abs(1-math.cos(math.radians(angle))))<=1.35*h_distance):
        return True
    else:
        return False

def process_image(img, file_name):
    
    pts_list = []
    len8_pts_list=[]
    angle_list=[]
    index=0
    dob_pts = None
        
    with open(r"C:/Users/ramsd/ML_Python/Aadhaar_Mask/CRAFT-pytorch/result/res_"+os.path.splitext(file_name)[0]+".txt", "r") as filestream:        
            for line in filestream:
                currentline = line.split(",")
                try:
                    a,b,c,d,e,f,g,h=currentline #Reads the bounding box coordinates from CRAFT txt file
                except:
                    continue #Skips blank lines in txt file

                a,b,c,d,e,f,g,h=int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h)

                points = np.array([[[a,b],[c,d],[e,f],[g,h]]])

                ar = aspect_ratio(a,b,c,d,e,f,g,h)

                if not ar_check(ar): #If crop doesn't fall in required dimensions, skip it
                    continue

                crop = crop_image(img, points) #Crops the image
                
                #Debug
#                 cv2.imshow('Crop', crop)
#                 cv2.waitKey(0)
                #Debug End
                
                if index<5: 
                    #Detects skew angle of first 5 crops, to give us overall skew of image. 
                    #We take mode of first 5 crops to avoid occasional wrong output
                    skew_angle = detect_angle(crop)
                    angle_list.append(round(skew_angle/10)*10)
                    angle, count = mode(angle_list)
                    rotation_angle = int(angle) if int(count)>2 else 0
                    index+=1
#                 else: #Takes angle of most recent first 5 crops for consistency
#                     skew_angle = detect_angle(crop)
#                     del angle_list[0]
#                     angle_list.append(round(skew_angle/10)*10)
#                     angle, count = mode(angle_list)
#                     rotation_angle = int(angle) if int(count)>2 else 0
                    
                    
                #Debug
#                 skew_angle = detect_angle(crop)
#                 print('Angle: ', skew_angle)
#                 print('Rotated Angle: ', rotation_angle)
#                 print('Angle List: ', angle_list)
                #Debug End
                    
                rotated = rotate_image(crop, rotation_angle)
                
                gray1, thresh1 = tesseract_preprocess(rotated, points, rotation_angle)#Preprocesses, giving us gray and threshold images
            
            #We need gray and thresh images bcuz in some cases tesseract is able to read thresh easier while in others
            #It is able to read grayscale images easier. Hence, we pass both through tesseract and compare generated strings
        
                #PSM 7 works better for rotated images while PSM 8 works better for non-rotated images
                if rotation_angle == 0:
                    string1 = pytesseract.image_to_string(gray1,lang='eng',config='--psm 8 --oem 3')
                    string2 = pytesseract.image_to_string(thresh1,lang='eng',config='--psm 8 --oem 3')
                else:
                    string1 = pytesseract.image_to_string(gray1,lang='eng',config='--psm 7 --oem 3')
                    string2 = pytesseract.image_to_string(thresh1,lang='eng',config='--psm 7 --oem 3')

                string = string1 if num_length(string1)>num_length(string2) else string2 #Choosing the better string of gray or thresh

                #Debug
#                 if num_length(string1)>num_length(string2):
#                     print(string1)
#                     cv2.imshow('Gray', gray1)
#                     cv2.waitKey(0)
#                 else:
#                     print(string2)
#                     cv2.imshow('Thresh',thresh1)
#                     cv2.waitKey(0)
                #Debug End

                length = num_length(string)

                if len(re.sub('[^0-9]','', string))<2: #Skip it if it has less than two detected numbers
                    continue

                if 11<=length<=13:
                    #Masks automatically if around 12 integers are present indicating Aadhaar no
                    mask_number(img, points)
                    
                elif length==8: #Masks all 8 len stuff except DOB (which has 2 slashes)
                    if len(re.sub('[^0-9/]','', string)) == 8:
                        mask_number(img, points)
                        
                elif 3<=length<12: #In other cases less than 12, it gets appended to list to compare with other nearby clusters of pts
                    pts_list.append(points)

                    #Debug
        #                 print(length, ", ",string)
        #                 cv2.imshow('Sidelined',prepped)
        #                 cv2.waitKey(0)
                    #Debug End

    for index, pts1 in enumerate(pts_list):
        if len(pts_list)==1:
            break
        for pts2 in pts_list[index+1:]:
            if neighbouring_boxes(pts1, pts2, rotation_angle):
                mask_number(img, pts1)
                mask_number(img, pts2)

                #Debug
#                     print(pts1[0][0][0], pts1[0][0][1], pts2[0][0][0], pts2[0][0][1])
#                     cv2.imshow('Masked', img)
#                     cv2.waitKey(0)
                #Debug End

        #Debug
#         cv2.imshow('Masked', img)
#         cv2.waitKey(0)
        #Debug End

    return img
    
if __name__ == '__main__':

    call_test(test_folder=args.test_folder,cuda=args.cuda) #Generating CRAFT Text Bounding Boxes
    
    print("CRAFT Box Detection Completed")
  
    #path_length=len(args.test_folder)
#   if (args.test_folder[-1]!='/'):
#     path_length=path_length+1
#   if (args.output_folder[-1]!='/'):
#     args.output_folder=args.output_folder+"/"    
#   paths=[]
#   for dirname, _, filenames in os.walk(args.test_folder):
#     for filename in filenames:
#         paths.append(os.path.join(dirname, filename))
#   try:
#     os.mkdir(args.output_folder)
#   except:
#     pass
    
    try:
        os.mkdir(args.output_folder) #Ensuring output directory exists
    except:
        pass
    
    if (args.test_folder[-1]!='/'):
        args.test_folder=args.test_folder+"/"
    if (args.output_folder[-1]!='/'):
        args.output_folder=args.output_folder+"/"
        
  
    pbar = tqdm(os.listdir(args.test_folder))
    
    for file_name in pbar:
        IMAGE_PATH = args.test_folder+file_name
        
        try:
            img = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
            height = img.shape[0]
            
        except:
            continue
            
        masked_img = process_image(img, file_name)
                                  
#     mean = np.mean(area_list)
#     std = np.std(area_list)
#     for i in range(len(num_list)):
#       (a,b,c,d,e,f,g,h) = num_list[i]
#       area = area_list[i]
#       if len(num_list)>3 and (((area-mean)/std)>2 or ((area-mean)/std)<-1):
#         continue
#       points = np.array([[[a,b],[c,d],[e,f],[g,h]]])
#       cv2.fillPoly(image, points, (0,0,0))
        
        cv2.imwrite(args.output_folder+"masked_"+file_name, masked_img)
        
    print("Masked Images Generated")

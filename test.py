"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    from collections import OrderedDict
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    if SHOW_TIME : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys

def call_test(test_folder=None,cuda=None):
    global TRAINED_MODEL
    global TEXT_THRESHOLD
    global LOW_TEXT
    global LINK_THRESHOLD
    global CUDA
    global CANVAS_SIZE
    global MAG_RATIO
    global POLY
    global SHOW_TIME
    global TEST_FOLDER
    global REFINE
    global REFINER_MODEL
    TRAINED_MODEL='model.pth'
    TEXT_THRESHOLD=0.7
    LOW_TEXT=0.4
    LINK_THRESHOLD=0.4
    CUDA=False
    CANVAS_SIZE=1280
    MAG_RATIO=1.5
    POLY=False
    SHOW_TIME=False
    TEST_FOLDER='../images/'
    REFINE=False
    REFINER_MODEL='weights/craft_refiner_CTW1500.pth'
    if test_folder:
        TEST_FOLDER=test_folder
    if cuda:
        CUDA=cuda

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(TEST_FOLDER)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

# load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + TRAINED_MODEL + ')')
    if CUDA:
        net.load_state_dict(copyStateDict(torch.load(TRAINED_MODEL)))
    else:
        net.load_state_dict(copyStateDict(torch.load(TRAINED_MODEL, map_location='cpu')))

    if CUDA:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if REFINE:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + REFINEr_model + ')')
        if CUDA:
            refine_net.load_state_dict(copyStateDict(torch.load(REFINEr_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(REFINEr_model, map_location='cpu')))

        refine_net.eval()
        POLY = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys = test_net(net, image, TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT, CUDA, POLY, refine_net)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))		


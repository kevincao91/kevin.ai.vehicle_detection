import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_beautiful
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

import numpy as np


def constraint_check(im_bgr, pascal_classes, scores, pred_boxes, thresh, class_agnostic):

    # print('hello!')
    im2show = np.copy(im_bgr)
    im_h = im2show.shape[0]
    im_w = im2show.shape[1]
    # 
    line_point_list = get_regional(im_h, im_w)
    im2show = regional_show(im2show, line_point_list)

    for j in range(1, len(pascal_classes)):
        # select classes
        if pascal_classes[j] not in ['bicycle', 'bus', 'car', 'cat', 'dog', 'motorbike', 'person']:
            continue
        # print(pascal_classes[j])
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            # regional_check
            cls_dets = regional_check(cls_dets.cpu().numpy(), line_point_list)
            # add boxes to img
            im2show = vis_detections_beautiful(im2show, pascal_classes[j], cls_dets, 0.5)

    return im2show


def get_regional(im_h, im_w):
    #                      |          \                     /
    #       -1             |           \                   /   
    #  ------------     -1 |  1      1  \   -1      1     /   -1
    #        1             |             \               /
    #                      |              \             /    
    # if y >= direction*(k*x+b):  keep
    # read point list
    # for img point
    
    # for video
    line_point_list = [( 0               , 0               , round(im_w*0.35), 0               ,  1),
                       ( round(im_w*0.35), 0               , im_w            , round(im_h*0.61),  1),
                       ( im_w            , round(im_h*0.61), im_w            , im_h            , -1),
                       ( im_w            , im_h            , round(im_w*0.26), im_h            , -1),
                       ( round(im_w*0.26), im_h            , 0               , round(im_h*0.61), -1),
                       ( 0               , round(im_h*0.61), 0               , 0               ,  1)]

    line_point_list = [( 0              , 0               , round(im_w*3/7), 0               ,  1),
                       ( round(im_w*3/7), 0               , im_w           , round(im_h*0.65),  1),
                       ( im_w           , round(im_h*0.65), im_w           , im_h            , -1),
                       ( im_w           , im_h            , round(im_w/3)  , im_h            , -1),
                       ( round(im_w/3)  , im_h            , 0              , round(im_h/3)   , -1),
                       ( 0              , round(im_h/3)   , 0              , 0               ,  1)]


    return line_point_list


def regional_show(im2show, line_point_list):
    
    # print('regional_show')
    color = (255,238,147)

    # 监测框显示
    for i in range(len(line_point_list)):
        cv2.line(im2show, line_point_list[i][0:2], line_point_list[i][2:4], color, thickness=6)

    return im2show


def regional_check(cls_dets, line_point_list):
    # 取每条线判断
    # print('init  ', 'line_point_list', len(line_point_list), 'cls_dets', len(cls_dets))
    for i in range(len(line_point_list)):
        x1, y1, x2, y2, direction = line_point_list[i]
        # 竖线
        if x2 == x1:
            keep_ind = []
            for j in range(cls_dets.shape[0]):
                bbox = tuple(int(np.round(x)) for x in cls_dets[j, :4])
                xx1, yy1, xx2, yy2 = bbox
                # calculate center point
                xx = round((xx1+xx2)/2)
                yy = round((yy1+yy2)/2)
                # point below line  -->  keep
                if xx >= direction*x1:
                    keep_ind.append(j)
            # 保留区域内点
            cls_dets = cls_dets[keep_ind]
            # print(i, 'cls_dets', len(cls_dets))
            continue
        # k = (y2-y1)/(x2-x1)
        k = (y2-y1) / (x2-x1)
        # b = y1-k*x1
        b = y1 - k * x1
        keep_ind = []
        for j in range(cls_dets.shape[0]):
            bbox = tuple(int(np.round(x)) for x in cls_dets[j, :4])
            xx1, yy1, xx2, yy2 = bbox
            # calculate center point
            xx = round((xx1+xx2)/2)
            yy = round((yy1+yy2)/2)
            # point below line  -->  keep
            if direction*yy >= direction*(k*xx+b):
                keep_ind.append(j)
        # 保留区域内点
        cls_dets = cls_dets[keep_ind]
        # print(i, 'cls_dets', len(cls_dets))
    pass
    return cls_dets






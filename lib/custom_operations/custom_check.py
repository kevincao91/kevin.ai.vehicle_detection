# --------------------------------------------------------
# PyTorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Kevin Cao, based on code from Jianwei Yang
# --------------------------------------------------------

import os
import sys
import numpy as np

import time
import cv2
import torch

from custom_operations.custom_show import vis_text_beautiful, vis_detections_beautiful


class CustomChecker:

    def __init__(self, regional_file_path):
        self.line_point_list = self.get_regional(regional_file_path)

    def get_regional(self, regional_file_path):
        #                      |          \                       /
        #       -1             |           \                     /   
        #  ------------     -1 |  1      1  \   -1       -1     /     1
        #        1             |             \                 /
        #                      |              \               /    
        # if direction*y >= direction*(k*x+b):  keep
        # read point list
        # for img point
        line_point_list = []

        with open(regional_file_path, 'r') as fh:
            lines = fh.readlines()
        
        im_w, im_h = int(lines[0].split()[0]), int(lines[0].split()[1])

        for line in lines[1:]:
            x1, y1, x2, y2, d = line.split()
            x1, y1, x2, y2, d = float(x1), float(y1), float(x2), float(y2), int(d)
            x1 = round(x1*im_w)
            y1 = round(y1*im_h)
            x2 = round(x2*im_w)
            y2 = round(y2*im_h)
            line_point_list.append((x1, y1, x2, y2, d))

        # for video
        # line_point_list = [( 0               , round(im_h*0.15), round(im_w*0.50), round(im_h*0.15),  1),
        #                    ( round(im_w*0.50), round(im_h*0.15), im_w            , round(im_h*0.61),  1),
        #                    ( im_w            , round(im_h*0.61), im_w            , im_h            , -1),
        #                    ( im_w            , im_h            , round(im_w*0.26), im_h            , -1),
        #                    ( round(im_w*0.26), im_h            , 0               , round(im_h*0.61), -1),
        #                    ( 0               , round(im_h*0.61), 0               , round(im_h*0.15),  1)]

        # for image
        # line_point_list = [( 0               , round(im_h*0.15), round(im_w*0.60), round(im_h*0.15),  1),
        #                    ( round(im_w*0.60), round(im_h*0.15), im_w            , round(im_h*0.65),  1),
        #                    ( im_w            , round(im_h*0.65), im_w            , im_h            , -1),
        #                    ( im_w            , im_h            , round(im_w/3)   , im_h            , -1),
        #                    ( round(im_w/3)   , im_h            , 0               , round(im_h/3)   , -1),
        #                    ( 0               , round(im_h/3)   , 0               , round(im_h*0.15),  1)]

        return line_point_list


    def constraint_check(self, im2show, all_cls_dets):

        # 
        line_point_list = self.line_point_list
        im2show = self.regional_show(im2show)

        if len(all_cls_dets):    # no value check
            for j in range(len(all_cls_dets)):
                cls_dets = all_cls_dets[j]
                cls_dets = self.regional_check(cls_dets)
                all_cls_dets[j] = cls_dets

        return im2show, all_cls_dets



    def regional_show(self, im2show):

        line_point_list = self.line_point_list

        # print('regional_show')
        color = (255, 238, 147)

        # 监测框显示
        for i in range(len(line_point_list)):
            cv2.line(im2show, line_point_list[i][0:2], line_point_list[i][2:4], color, thickness=6)

        return im2show


    def regional_check(self, cls_dets):
        
        line_point_list = self.line_point_list
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






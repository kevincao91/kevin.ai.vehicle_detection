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
        self.smooth = False
        self.identify = True
        self.all_cls_dets_time_seq = []
        self.idf_center_list = [[[0, 0, 0], ]]
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


    def regional_check(self, im2show, all_cls_dets):

        # 
        line_point_list = self.line_point_list
        im2show = self.regional_show(im2show)

        for j in range(len(all_cls_dets)):
            cls_dets = all_cls_dets[j]
            if cls_dets.any():    # no value check
                cls_dets = self.get_in_regional(cls_dets)
                all_cls_dets[j] = cls_dets

        return im2show, all_cls_dets

    def identify_check(self, im2show, all_cls_dets):

        point_size = 1
        color1 = (0, 0, 255) # BGR
        color2 = (0, 255, 0) # BGR
        color3 = (255, 0, 0) # BGR
        thickness = 4 # 可以为 0 、4、8
        f_s = cv2.FONT_HERSHEY_SIMPLEX  # 文字字体
        t_s = 0.8  # 文字大小
        t_t = 2    # 文字线条粗细

        # get center by class
        for j in range(len(all_cls_dets)):
            print('cls:', j)
            cls_dets = all_cls_dets[j]
            if cls_dets.any():    # no value check
                cls_dets_center = self.get_rec_center(cls_dets)
                # plot det center
                for dets_center in cls_dets_center:
                    cv2.circle(im2show, dets_center, point_size, color1, thickness)
            # plot idf center
            cls_idf_center = self.idf_center_list[j]
            for idf_center in cls_idf_center:
                idf, xx, yy = idf_center
                cv2.circle(im2show, (xx, yy), 5, color2, thickness)
            # identify label
            labels = self.identify_label(j, cls_dets_center)
            #
            for i, label in enumerate(labels):
                if label > 0:
                    # plot label
                    # cv2.circle(im2show, cls_dets_center[i], 8, color3, thickness)
                    cv2.putText(im2show, str(label), cls_dets_center[i], f_s, t_s, color3, thickness=1)
        pass
        return im2show, all_cls_dets

    def identify_label(self, cls_idx, cls_dets_center):
        cls_idf_center = self.idf_center_list[cls_idx]
        labels = np.zeros((len(cls_dets_center),), dtype=np.int)
        print(labels)
        for idf_center in cls_idf_center:
            print('idf_center', idf_center)
            idf, x_c, y_c = idf_center
            for i, det_center in enumerate(cls_dets_center):
                print('cls_dets_center  ', i, det_center)
                xx, yy = det_center
                dist = (xx-x_c)**2 + (yy-y_c)**2
                print('dist : ', dist)
                if dist < 100:
                    # get exist label
                    labels[i] = idf
        # add new label
        for i, label in enumerate(labels):
            xx, yy = cls_dets_center[i]
            if label == 0:
                new_num = cls_idf_center[-1][0] + 1
                cls_idf_center.append([new_num, xx, yy])
                labels[i] = new_num
        self.idf_center_list[cls_idx] = cls_idf_center
        print(labels)
        print(self.idf_center_list[cls_idx])
        pass
        return labels

    def get_rec_center(self, cls_dets):
        cls_dets_center = []
        for j in range(cls_dets.shape[0]):
            bbox = tuple(int(np.round(x)) for x in cls_dets[j, :4])
            xx1, yy1, xx2, yy2 = bbox
            x_c = round((xx1 + xx2) / 2)
            y_c = round((yy1 + yy2) / 2)
            cls_dets_center.append((x_c, y_c))
        pass
        
        return cls_dets_center

    def regional_show(self, im2show):

        line_point_list = self.line_point_list

        # print('regional_show')
        color = (255, 238, 147)

        # 监测框显示
        for i in range(len(line_point_list)):
            cv2.line(im2show, line_point_list[i][0:2], line_point_list[i][2:4], color, thickness=6)

        return im2show


    def get_in_regional(self, cls_dets):
        
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






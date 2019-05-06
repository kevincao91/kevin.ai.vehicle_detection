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
        self.dist_th = 1000
        self.frame_rate = 24
        self.center_keep_time = self.frame_rate
        self.path_keep_time = self.frame_rate
        self.path_speed_dict = {}
        self.idf_center_list = [[[0, 0, 0, 0, 0.0], ]]   # idf, x_c, y_c, keep_time, speed
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
        color3 = (0, 0, 255) # BGR
        thickness = 2 # 可以为 0 、4、8
        f_s = cv2.FONT_HERSHEY_SIMPLEX  # 文字字体
        t_s = 0.6  # 文字大小
        t_t = 2    # 文字线条粗细

        all_cls_labels = []
        all_cls_speeds = []
        # get center by class
        for j in range(len(all_cls_dets)):
            # print('cls:', j)
            cls_dets = all_cls_dets[j]
            if cls_dets.any():    # no value check
                cls_dets_center = self.get_rec_center(cls_dets)
                # plot det center
                for dets_center in cls_dets_center:
                    cv2.circle(im2show, dets_center, point_size, color1, thickness)
            # plot idf center
            cls_idf_center = self.idf_center_list[j]
            for idf_center in cls_idf_center:
                idf, xx, yy, count, speed = idf_center
                if idf and count:
                    cv2.circle(im2show, (xx, yy), 3, color2, thickness)
            # identify label
            labels, speeds = self.identify_label(j, cls_dets_center)  # UnboundLocalError: local variable 'cls_dets_center' referenced before assignment
            all_cls_labels.append(labels)
            all_cls_speeds.append(speeds)
        # move path save
        self.path_save()
        # plot path line
        im2show = self.path_show(im2show)
        pass
        # plot labels
        # for j in range(len(all_cls_dets)):
        #     cls_labels = all_cls_labels[j]
        #     for i, label in enumerate(cls_labels):
        #         if label > 0:
        #             # plot label
        #             # cv2.circle(im2show, cls_dets_center[i], 8, color3, thickness)
        #             cv2.putText(im2show, str(label), cls_dets_center[i], f_s, t_s, color3, thickness=3)
        pass
        return im2show, all_cls_dets, all_cls_labels, all_cls_speeds

    def path_show(self, im2show):
    
        speed_color_list = [(200, 200, 0), (0, 200, 200), (0, 0, 200), (0, 0, 255)]
        # print('show ====>', self.path_speed_dict)
        for key in self.path_speed_dict.keys():
            path_speed_list = self.path_speed_dict[key]
            if len(path_speed_list) < 2:
                continue
            for i in range(len(path_speed_list)-1):
                x1, y1, speed1 = path_speed_list[i]
                point1 = (x1, y1)
                x2, y2, speed2 = path_speed_list[i+1]
                point2 = (x2, y2)
                speed_avg = (speed1 + speed2) / 2
                if speed_avg <= 2.0:
                    color = speed_color_list[0]
                elif speed_avg <= 4.0:
                    color = speed_color_list[1]
                elif speed_avg <= 6.0:
                    color = speed_color_list[2]
                else:
                    color = speed_color_list[3]
                cv2.line(im2show, point1, point2, color, thickness=3)
        
        return im2show


    def path_save(self):
        # print('befor save ====>', self.path_speed_dict)
        for idx_cls in range(len(self.idf_center_list)):
            cls_idf_center = self.idf_center_list[idx_cls]
            # print('cls_idf_center', cls_idf_center)
            for idf_center in cls_idf_center:
                idf, xx, yy, count, speed = idf_center
                # print(idf, xx, yy, count, speed)
                if not idf:     # not zero label
                    continue
                if count:     # is avaliable
                    # save path
                    if idf in self.path_speed_dict.keys():
                        # print('exist')
                        path_speed_list = self.path_speed_dict[idf]
                        path_speed_list.append((xx, yy, speed))
                        # long limit
                        if len(path_speed_list) > self.path_keep_time:
                            del path_speed_list[0]
                        self.path_speed_dict[idf] = path_speed_list
                    else:
                        # print('not exist')
                        self.path_speed_dict[idf] = [(xx, yy, speed)]
                else:    # not avaliable
                    # del path
                    if idf in self.path_speed_dict.keys():
                        # print('exist not avalible key %d !' % idf)
                        del self.path_speed_dict[idf]
                
                # print(self.path_speed_dict)
        # print('after save ====>', self.path_speed_dict)

        pass


    def identify_label(self, cls_idx, cls_dets_center):
        cls_idf_center = self.idf_center_list[cls_idx]
        labels = np.zeros((len(cls_dets_center),), dtype=np.int)
        speeds = np.zeros((len(cls_dets_center),), dtype=np.float32)
        # print('initial labels: ', labels)
        for idf_idx, idf_center in enumerate(cls_idf_center):
            # print('idf_center', idf_center)
            idf, x_c, y_c, count, speed = idf_center
            if count<=0 or idf==0:   # check label is avalible
                # print('continue!')
                continue
            best_det_idx = -1
            best_det_dist = 100000000
            for i, det_center in enumerate(cls_dets_center):
                # print('cls_dets_center  ', i, det_center)
                xx, yy = det_center
                dist = (xx-x_c)**2 + (yy-y_c)**2
                # print('dist : ', dist)
                if dist < best_det_dist:
                    best_det_dist = dist
                    best_det_idx = i
            # print('best_det_dist', best_det_dist)
            # print('best_det_idx', best_det_idx)
            # 
            if best_det_dist < self.dist_th:
                # get exist label and speed
                labels[best_det_idx] = idf
                speed = best_det_dist / self.frame_rate
                speeds[best_det_idx] = speed
                xx, yy = cls_dets_center[best_det_idx]
                x_c = xx
                y_c = yy
                # print('find ', 'cls_dets_center  ', best_det_idx, cls_dets_center[best_det_idx], 'like ', 'idf_center', idf_center)
            # print('after label %d modifity labels: ' % idf, labels)
            #
            if idf in labels:
                # print('idf_center ', idf, '+')
                if count < self.center_keep_time:
                    count += 1
            else:
                # print('idf_center ', idf, '-')
                if count > 0:
                    count -= 1
            # update 
            cls_idf_center[idf_idx] = [idf, x_c, y_c, count, speed]
            # print('update ', 'cls_idf_center  ', [idf, x_c, y_c, count])
        # update
        self.idf_center_list[cls_idx] = cls_idf_center
        # add new label
        for i, label in enumerate(labels):
            xx, yy = cls_dets_center[i]
            if label == 0:
                new_num = cls_idf_center[-1][0] + 1
                cls_idf_center.append([new_num, xx, yy, self.center_keep_time, 0.0])
                labels[i] = new_num
        self.idf_center_list[cls_idx] = cls_idf_center
        # print('finally modifity labels: ', labels)
        # print('finally modifity cls_idf_center: ', self.idf_center_list[cls_idx])
        pass
        return labels, speeds

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






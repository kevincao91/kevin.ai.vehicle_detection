# --------------------------------------------------------
# PyTorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Kevin Cao
# --------------------------------------------------------


import cv2
import numpy as np


def vis_detections_beautiful(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    im_w = im.shape[0]
    im_h = im.shape[1]
    show_scale = 1.0 * np.e ** ((min(im_w, im_h) - 1000) / 1000)
    for i in range(np.minimum(20, dets.shape[0]) - 1, -1, -1):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            x1, y1, x2, y2 = bbox
            color1, color2, color3 = (76, 180, 231), (255, 192, 159), (76, 180, 231)
            color4, color5 = (0, 0, 0), (0, 0, 0)

            # 关于font_face参数：
            #
            # FONT_HERSHEY_SIMPLEX           -   正常大小无衬线字体. 
            # FONT_HERSHEY_PLAIN             -   小号无衬线字体.
            # FONT_HERSHEY_DUPLEX            -   更复杂的正常大小无衬线字体.
            # FONT_HERSHEY_COMPLEX           -   正常大小有衬线字体.
            # FONT_HERSHEY_TRIPLEX           -   正常大小有衬线字体(比CV_FONT_HERSHEY_COMPLEX更复杂) 
            # FONT_HERSHEY_COMPLEX_SMALL     -   CV_FONT_HERSHEY_COMPLEX的小一版.
            # FONT_HERSHEY_SCRIPT_SIMPLEX    -   手写风格字体.
            # FONT_HERSHEY_SCRIPT_COMPLEX    -   更复杂的手写风格字体.

            f_s = cv2.FONT_HERSHEY_SIMPLEX  # 文字字体
            t_s = 0.6 * show_scale  # 文字大小
            t_t = round(1.5 * show_scale)  # 文字线条粗细
            t_h = round(6 * show_scale)  # 文字高度
            r_h = round(25 * show_scale)  # 背景矩形高度
            r_w = round(70 * show_scale)  # 背景矩形宽度
            r_s = round(2.5 * show_scale)  # 选框边界大小
            # 小框
            cv2.rectangle(im, (x2, y1 - round(r_s / 2)), (x2 + r_w, y1 + r_h), color1, thickness=-1)
            cv2.rectangle(im, (x2, y1 - round(r_s / 2) + r_h), (x2 + r_w, y1 + 2 * r_h), color2, thickness=-1)
            # 大框
            cv2.rectangle(im, (x1, y1), (x2, y2), color3, thickness=r_s)
            # 文字
            cv2.putText(im, '%s' % class_name, (x2 + r_s, y1 - round(r_s / 2) + round(r_h / 2) + t_h), f_s, t_s, color4,
                        thickness=t_t)
            cv2.putText(im, '%.3f' % score, (x2 + r_s, y1 - round(r_s / 2) + round(r_h * 3 / 2) + t_h), f_s, t_s,
                        color5, thickness=t_t)

    return im


def vis_text_beautiful(im, str_list):
    """Visual debugging of detections."""
    im_h = im.shape[0]
    im_w = im.shape[1]

    show_scale = 1.0 * np.e ** ((min(im_w, im_h) - 1000) / 1000)
    # print(show_scale)

    gpu_name, mem_used, mem_total, model_name, file_name, detect_time_avg, nms_time_avg, total_time_avg,\
        frame_rate_avg = str_list

    file_name = file_name.split('/')[1] + '/' + file_name.split('/')[3][:-4]

    show_string_1 = 'device: %s  mem: %.3f G / %.3f G  model: %s  size: %d * %d' % (gpu_name, mem_used, mem_total,
file_name, im_w, im_h)
    show_string_2 = 'detect: %.3fs    nms: %.3fs    total: %.3fs    FPS: %.3f' % (detect_time_avg, nms_time_avg,
                                                                                  total_time_avg, frame_rate_avg)

    color1, color2 = (255, 255, 255), (255, 255, 255)

    f_s = cv2.FONT_HERSHEY_SIMPLEX  # 文字字体
    t_s = 1.0 * show_scale  # 文字大小
    t_t = round(2 * show_scale)  # 文字线条粗细

    # location
    x1, y1 = 20, 30
    x2, y2 = 20, 60

    # 文字
    cv2.putText(im, show_string_1, (x1, y1), f_s, t_s, color1, thickness=t_t)
    cv2.putText(im, show_string_2, (x2, y2), f_s, t_s, color2, thickness=t_t)

    return im

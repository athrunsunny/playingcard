# -*- coding: utf-8 -*-
"""
Time:     2021.10.26
Author:   Athrunsunny
Version:  V 0.1
File:     inference.py
Describe: Functions in this file is use to inference
"""

import cv2
import torch
import time
import onnxruntime
import numpy as np
from function.utils import LoadImages, Annotator, colors, check_img_size, non_max_suppression, scale_coords
from function import config as CFG


def load_model(weights, **options):
    imgsz = options.pop('imgsz', 640)
    stride = options.pop('stride', 64)

    w = str(weights[0] if isinstance(weights, list) else weights)
    session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)
    return session, imgsz, stride


def image_process(img):
    assert isinstance(img, np.ndarray)
    img = img.astype('float32')
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]
    return img


def inference(session, img, **options):
    conf_thres = options.pop('conf_thres', 0.25)
    iou_thres = options.pop('iou_thres', 0.45)
    classes = options.pop('classes', None)
    agnostic = options.pop('agnostic', False)
    max_det = options.pop('max_det', 1000)

    pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, max_det=max_det,
                               agnostic=agnostic)
    return pred


def post_process(pred, img, im0s, dataset, **options):
    showImg = options.pop('showImg', False)
    hide_conf = options.pop('hide_conf', False)
    hide_labels = options.pop('hide_labels', False)
    line_thickness = options.pop('line_thickness', 1)
    labelDict = options.pop('labelDict', None)

    labels = labelDict['cardlabel']
    res_label = []
    for i, det in enumerate(pred):
        s, im0, frame = '', im0s.copy(), getattr(dataset, 'frame', 0)
        annotator = Annotator(im0, line_width=line_thickness, example=str(labels))
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = None if hide_labels else (labels[c] if hide_conf else f'{labels[c]} {conf:.2f}')
                label_no_conf = None if hide_labels else (labels[c] if hide_conf else f'{labels[c]}')
                res_label.append(label_no_conf)
                annotator.box_label(xyxy, label, color=colors(c, True))
        print(f'{s}')
        im0 = annotator.result()
        if showImg:
            cv2.imshow('result', im0)
            cv2.waitKey(0)
    return res_label


def run(weights, source, **options):
    conf_thres = options.pop('conf_thres', 0.25)  # confidence threshold
    iou_thres = options.pop('iou_thres', 0.45)  # NMS IOU threshold
    classes = options.pop('classes', None)  # filter by class: --class 0, or --class 0 2 3
    agnostic = options.pop('agnostic', False)  # class-agnostic NMS
    max_det = options.pop('max_det', 1000)  # maximum detections per image
    hide_conf = options.pop('hide_conf', False)  # hide confidences
    hide_labels = options.pop('hide_labels', False)  # hide labels
    line_thickness = options.pop('line_thickness', 1)  # bounding box thickness (pixels)
    imgsz = options.pop('imgsz', 640)  # inference size (pixels)
    showImg = options.pop('showImg', False)  # show results
    labelDict = options.pop('labelDict', CFG.LABEL_DICT)  # config labels

    session, imgsz, stride = load_model(weights=weights, imgsz=imgsz)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
    res = []
    for path, img, im0s, vid_cap in dataset:
        img = image_process(img)
        t1 = time.time()
        pred = inference(session, img, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, classes=classes,
                         agnostic=agnostic)
        t2 = time.time()
        print('Inference time:%.3fs' % (t2 - t1))
        res = post_process(pred, img, im0s, dataset, hide_conf=hide_conf, hide_labels=hide_labels,
                           line_thickness=line_thickness, showImg=showImg, labelDict=labelDict)
    return res


if __name__ == '__main__':
    imagepath = 'image/1.jpg'
    modelpath = 'model/weight.onnx'
    res = run(modelpath, imagepath, showImg=True)
    print(res)
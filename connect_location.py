import os.path
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/pythonProject/yolov5'))))
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


WEIGHTS = 'yolov5s.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

def detect(num,FRAME):
    frame, weights, imgsz = FRAME, WEIGHTS, IMG_SIZE

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Load image
    img0 = frame  # BGR
    assert img0 is not None, 'Image Not Found ' + frame

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]
    print('pred shape:', pred.shape)

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]
    print('det shape:', det.shape)

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        max = 0
        save_location = []
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            if max <= float(f'{conf:.2f}'):
                save_location = xyxy
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
        if num:
            return save_location
        print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')
        # if num:
        #     return det
    # Stream result
    # print(s)
    # cv2.imshow(source, img0)
    # cv2.waitKey(0)  # 1 millisecond


if __name__ == '__main__':
    source = '/Users/maxcha/PycharmProjects/pythonProject/HumanBody-Skeleton-Detection-using-OpenCV/ss.jpeg'
    frame = cv2.imread(source)
    #check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
            detect(1,frame)
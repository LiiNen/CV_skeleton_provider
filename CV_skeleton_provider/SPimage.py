import argparse
import cv2
import imutils
import time
import numpy as np
import sys,os

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils_formatter import optionChecker
# from connect_location import detect
from utils_formatter import optionChecker
from utils_preprocessor import preBack
from utils_preprocessor import preGray
from utils_preprocessor import preGamma
from utils_preprocessor import preBlackProportion

def forImage(opt):
    print('img')
    source, out_path, option, exclude, weightsFile, protoFile, threshold, gray_bool, back_bool, selectRect_bool, auto_bool, gamma_value, b_propo_bool = \
      opt.source, opt.output, opt.option, opt.exclude, opt.weight, opt.proto, opt.thres, opt.gray, opt.back, opt.selectRect, opt.autolocation, opt.gamma, opt.b_propo
    
    opt_dict = optionChecker(option)

    if exclude != -1:
        for ex_point in exclude:
            if ex_point < 0 or ex_point > 17:
                print('exclude points out of range.')
                return

    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    frame = cv2.imread(source)
    originFrame = frame.copy()
    
    if gamma_value > 0:
        frame = preGamma(frame, gamma_value)
    if b_propo_bool:
        preBlackPropotion(frame)
    if back_bool:
        rect_init = (0, 0, 0, 0)
        if auto_bool:
          #temp = detect(1, frame)
          #temp_rect = (int(temp[0]), int(temp[1]), int(temp[2])-10, int(temp[3]-10))
          #frame, unused_rect = preBack(frame, selectRect_bool, temp_rect)
          print('imported none')
        frame = preBack(frame, selectRect_bool, rect_init)
    if gray_bool:
        frame = preGray(frame, source)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # 네트워크 인풋 사이즈 설정
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("time taken : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원본이미지 좌표에 대입
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        # threshold 넘는 것만 keypoint 저장
        if prob > threshold:
            points.append((int(x), int(y)))
            if (opt_dict['keyp']):
                cv2.circle(originFrame, points[-1], 8, (0, 255, 255),
                           thickness=-1, lineType=cv2.FILLED)
            if (opt_dict['label']):
                cv2.putText(originFrame, "{}".format(i), points[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
        else:
            points.append(None)

    # skeleton 구조 연결해주기
    if (opt_dict['skel']):
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(originFrame, points[partA], points[partB], (0, 255, 255), 2)

    cv2.imshow('output', frame)
    cv2.imshow('output_origin', originFrame)
    cv2.imwrite(out_path + '.jpg', originFrame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)
    return

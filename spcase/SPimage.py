import argparse
import cv2
import time
import numpy as np
import sys,os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.formatter import optionChecker
from connect_location import detect

def forImage(opt):
    print('img')
    source, out_path, option, exclude, weightsFile, protoFile, threshold, gray_bool, back_bool, selectRect_bool, auto_location = opt.source, opt.output, opt.option, opt.exclude, opt.weight, opt.proto, opt.thres, opt.gray, opt.back, opt.selectRect, opt.autolocation
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
    if back_bool:
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        if selectRect_bool:
            rect = cv2.selectROI(frame)
        elif auto_location:
            temp = detect(1,frame)
            rect = (int(temp[0]),int(temp[1]),int(temp[2])-10,int(temp[3]-10))
        else:
            rect = (10, 10, frame.shape[1] - 10, frame.shape[0] - 10)
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        frame = frame * mask2[:, :, np.newaxis]
    if gray_bool:
        frame = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        bgr = frame[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # alpha = rgb2gray(frame)  # Channel 3
        frame = np.dstack([bgr])  # Add the alpha channel

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
                cv2.circle(frame, points[-1], 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            if (opt_dict['label']):
                cv2.putText(frame, "{}".format(i), points[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
        else:
            points.append(None)

    # skeleton 구조 연결해주기
    if (opt_dict['skel']):
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)

    cv2.imshow('output', frame)
    cv2.imwrite(out_path + '.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)
    return
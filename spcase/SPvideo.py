import argparse
import cv2
import time
import numpy as np
import math

from utils.formatter import optionChecker

def forVideo(opt):
    print('video')
    source, option, exclude, weightsFile, protoFile, threshold, out_path, comp = opt.source, opt.option, opt.exclude, opt.weight, opt.proto, opt.thres, opt.output, opt.comp
    opt_dict = optionChecker(option)
    if exclude != -1:
        for ex_point in exclude:
            if ex_point < 0 or ex_point > 17:
                print('exclude points out of range.')
                return

    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
    frames = cv2.VideoCapture(source)
    # cv2.VideoCapture.get(5) : 영상의 fps를 가져오는 함수+Id, return float
    # comp로 나눈 프레임값을 사용
    sourceFps = frames.get(5)
    outputFps = int(sourceFps / comp)

    frameLeft, frame = frames.read()
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # output 비디오 쓰기
    videoWriter = cv2.VideoWriter(out_path + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), outputFps, (frameWidth, frameHeight))
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    inWidth = 368
    inHeight = 368
    t = time.time()
    while(frameLeft):
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        # keypoint 저장
        points = []
        for i in range(nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # 원본이미지 좌표에 대입
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            # threshold 넘는 것만 keypoint 저장
            if prob > threshold : 
                points.append((int(x), int(y)))
                if(opt_dict['keyp']):
                    cv2.circle(frame, points[-1], 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                if(opt_dict['label']):
                    cv2.putText(frame, "{}".format(i), points[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            else :
                points.append(None)

        # skeleton 구조 연결해주기
        if(opt_dict['skel']):
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        videoWriter.write(frame)

        # 프레임 건너뛰기
        # source fps가 comp로 나누어떨어지지 않아도 비슷한 영상 길이(시간)을 제공
        for i in range(0, int(sourceFps/outputFps)):
            frameLeft, frame = frames.read()
        cv2.waitKey(outputFps)

    videoWriter.release()
    print("time taken : {:.3f}".format(time.time() - t))
    return
import argparse
import cv2
import imutils
import time
import numpy as np


# def deleteShadow(frame):
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     out = cv2.adaptiveThreshold(
#         frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
#     out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
#     return out


# def hogDetector(frame):
#     HOGCV = cv2.HOGDescriptor()
#     HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#     bounding_box_cordinates, weights = HOGCV.detectMultiScale(
#         frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

#     person = 1
#     for x, y, w, h in bounding_box_cordinates:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, f'person {person}', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#         person += 1

#     cv2.putText(frame, 'Status : Detecting ', (40, 40),
#                 cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
#     cv2.putText(frame, f'Total Persons : {person-1}',
#                 (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
#     cv2.imshow('output', frame)

#     return frame


# # 의미 없음
# def flipColor(frame):
#     return 255-frame

def findWhite(frame):
    # ROI세팅 - 사람이 존재하는 영역 찾기
    # area=width*height
    # lower_white = np.array([0, 0, 0], dtype=np.uint8)
    # upper_white = np.array([0, 0, 255], dtype=np.uint8)
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    # mask_gray = cv2.inRange(frame, lower_white, upper_white)
    # img_res = cv2.bitwise_and(frame, frame, mask=mask_gray)
    # Hue is from 0-179 for Opencv
    # Set minimum and max HSV values to display
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    output = np.array(output)
    h, w = output.shape[:2]
    # Get list of unique colours...
    # Arrange all pixels into a tall column of 3 RGB values and find unique rows (colours)
    colours, counts = np.unique(
        output.reshape(-1, 3), axis=0, return_counts=1)
    # # print(colours[0])
    # black_proportion = (100*counts[0])/(h*w)
    # print(black_proportion)

    # # Iterate through unique colours
    for index, colour in enumerate(colours):
        if(colour[0] == 0 & colour[1] == 0 & colour[2] == 0):
            cv2.imwrite('color.jpg', output)
            return (100*counts[index])/(h*w)


def gamma(frame):
    g = float(input("감마 값 : "))
    out = frame.copy()
    out = frame.astype(np.float)
    out = ((out / 255) ** (1 / g)) * 255
    out = out.astype(np.uint8)
    # out = hogDetector(out)
    return out


def equalize(frame):
    image_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)  # YUV로 변경합니다.
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])  # 히스토그램 평활화를 적용
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return image_rgb


def forImage(opt):
    print('img')
    source, skeleton_bool, keypoint_bool, exclude, weightsFile, protoFile, threshold = opt.source, opt.skel, opt.keyp, opt.exclude, opt.weight, opt.proto, opt.thres

    if exclude != -1:
        for ex_point in exclude:
            if ex_point < 0 or ex_point > 17:
                print('exclude points out of range.')
                return

    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [
        8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    frame = cv2.imread(source)
    frame = gamma(frame)
    # frame = flipColor(frame)
    # frame = imutils.resize(frame, width=min(800, frame.shape[1]))
    # frame = equalize(frame)
    # frame = deleteShadow(frame)
    black_proportion = findWhite(frame)
    # if(black_proportion < 30):
    frame = equalize(frame)
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # 네트워크 인풋 사이즈 설정
    inWidth = 368
    inHeight = 368
    # inWidth = 720
    # inHeight = 720
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

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
        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255),
                       thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(
                y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            points.append((int(x), int(y)))
        else:
            points.append(None)

    # 원본 이미지 위에 그리기
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255),
                       thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('output_keypoints', frameCopy)
    cv2.imshow('output_skeleton', frame)

    cv2.imwrite('output_keypoints.jpg', frameCopy)
    cv2.imwrite('output_skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)
    return

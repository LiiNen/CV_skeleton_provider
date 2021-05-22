import argparse
import cv2
import time
import numpy as np

def forImage(opt):
    source, skeleton_bool, keypoint_bool, exclude, weightsFile, protoFile, threshold = opt.source, opt.skel, opt.keyp, opt.exclude, opt.weight, opt.proto, opt.thres

    if exclude != -1:
        for ex_point in exclude:
            if ex_point < 0 or ex_point > 17:
                print('exclude points out of range.')
                return

    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    if source.split('.')[1] in ['jpg', 'jpeg', 'png']:
        print('img')
    elif source.split('.')[1] in ['mp4', 'avi', 'mkv']:
        print('video')
        return
    else:
        print('source file error')
        return

    frame = cv2.imread(source)
    frameCopy = np.copy(frame)
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
        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            points.append((int(x), int(y)))
        else :
            points.append(None)

    # 원본 이미지 위에 그리기
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    cv2.imshow('output_keypoints', frameCopy)
    cv2.imshow('output_skeleton', frame)

    cv2.imwrite('output_keypoints.jpg', frameCopy)
    cv2.imwrite('output_skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)
    return

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='example.jpeg', help='input source path. default example.jpeg')
    parser.add_argument('--skel', type=str2bool, default=True, help='if you want not to draw skeleton, set arg to False')
    parser.add_argument('--keyp', type=str2bool, default=True, help='if you want not to draw ketpoints, set arg to False')
    parser.add_argument('--exclude', nargs='+', type=int, default=-1, help='points to exclude. args for spacing. -1 for none(default), 0~17 to exclude. else error')
    parser.add_argument('--proto', type=str, default='pose/coco/pose_deploy_linevec.prototxt', help='for model. default pose/coco/pose_deploy_linevec.prototxt')
    parser.add_argument('--weight', type=str, default='pose/coco/pose_iter_440000.caffemodel', help='for model. default pose/coco/pose_iter_440000.caffemodel')
    parser.add_argument('--thres', type=float, default=0.1, help='set threshold for detecting. default 0.1')
    opt = parser.parse_args()
    print(opt)

    forImage(opt)
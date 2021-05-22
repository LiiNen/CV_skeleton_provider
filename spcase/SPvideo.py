import argparse
import cv2
import time
import numpy as np

def forVideo(opt):
    print('video')
    source, skeleton_bool, keypoint_bool, exclude, weightsFile, protoFile, threshold = opt.source, opt.skel, opt.keyp, opt.exclude, opt.weight, opt.proto, opt.thres

    if exclude != -1:
        for ex_point in exclude:
            if ex_point < 0 or ex_point > 17:
                print('exclude points out of range.')
                return

    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]   
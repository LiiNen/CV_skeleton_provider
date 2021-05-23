import argparse
import cv2
import time
import numpy as np

from spcase.SPimage import forImage
from spcase.SPvideo import forVideo
from utils.formatter import str2bool
from utils.formatter import fileformat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='example.jpeg', help='input source path. default example.jpeg')
    parser.add_argument('--output', type=str, default='output', help='output source path, until filename(format exclude). default output')
    parser.add_argument('--skel', type=str2bool, default=True, help='if you want not to draw skeleton, set arg to False')
    parser.add_argument('--keyp', type=str2bool, default=True, help='if you want not to draw ketpoints, set arg to False')
    parser.add_argument('--exclude', nargs='+', type=int, default=-1, help='points to exclude. args for spacing. -1 for none(default), 0~17 to exclude. else error')
    parser.add_argument('--proto', type=str, default='pose/coco/pose_deploy_linevec.prototxt', help='for model. default pose/coco/pose_deploy_linevec.prototxt')
    parser.add_argument('--weight', type=str, default='pose/coco/pose_iter_440000.caffemodel', help='for model. default pose/coco/pose_iter_440000.caffemodel')
    parser.add_argument('--thres', type=float, default=0.1, help='set threshold for detecting. default 0.1')
    parser.add_argument('--gray', type=str2bool, default=False, help='preprocessing using gray img, set True')
    parser.add_argument('--comp', type=int, default=1, help='reducing fps only for video. fps/comp. default 1')

    opt = parser.parse_args()
    print(opt)
    
    if fileformat(opt.source) == 0:
        forImage(opt)
    elif fileformat(opt.source) == 1:
        forVideo(opt)
    else:
        print('source file error')
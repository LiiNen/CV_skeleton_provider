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
    
    parser.add_argument('--source', type=str, default='../example.jpeg', help='input source path. default example.jpeg')
    parser.add_argument('--output', type=str, default='../output', help='output source path, until filename(format exclude). default output')
    parser.add_argument('--option', type=str, default='skl', help='draw option. contains s, k, l skeleton, keypoint, label respectively. default skl')
    parser.add_argument('--exclude', nargs='+', type=int, default=-1, help='points to exclude. args for spacing. -1 for none(default), 0~17 to exclude. else error')
    parser.add_argument('--proto', type=str, default='./pose/coco/pose_deploy_linevec.prototxt', help='for model. default pose/coco/pose_deploy_linevec.prototxt')
    parser.add_argument('--weight', type=str, default='./pose/coco/pose_iter_440000.caffemodel', help='for model. default pose/coco/pose_iter_440000.caffemodel')
    parser.add_argument('--thres', type=float, default=0.1, help='set threshold for detecting. default 0.1')
    parser.add_argument('--gray', type=str2bool, default=False, help='preprocessing using gray img, set True')
    parser.add_argument('--back', type=str2bool, default=False, help='preprocessing removing background img, set True')
    parser.add_argument('--selectRect', type=str2bool, default=False, help='preprocessing select Rect to masking removed background img')
    parser.add_argument('--autolocation', type=str2bool, default=False, help='preprocessing using auto location to check boundary, set True')
    parser.add_argument('--comp', type=int, default=1, help='reducing fps only for video. fps/comp. default 1')
    parser.add_argument('--gamma', type=float, default=-1, help='gamma over 0. default -1(none)')
    parser.add_argument('--b_propo', type=str2bool, default=False, help='black proportion')
    
    opt = parser.parse_args()
    print(opt)
    
    if fileformat(opt.source) == 0:
        forImage(opt)
    elif fileformat(opt.source) == 1:
        forVideo(opt)
    else:
        print('source file error')

def defaultDict():
    default_dict = {}
    default_dict['--source'] = './example.jpeg'
    default_dict['--output'] = './output'
    default_dict['--option'] = 'skl'
    default_dict['--exclude'] = []
    default_dict['--proto'] = './pose/coco/pose_deploy_linevec.prototxt'
    default_dict['--weight'] = './pose/coco/pose_iter_440000.caffemodel'
    default_dict['--thres'] = 0.1
    default_dict['--gray'] = False
    default_dict['--back'] = False
    default_dict['--selectRect'] = False
    default_dict['--autolocation'] = False
    default_dict['--comp'] = 1
    default_dict['--gamma'] = -1
    default_dict['--b_propo'] = False
    return default_dict

def skprovider(dict_object):
    arg_list = []
    keys = dict_object.keys()
    for key in keys:
        if key == '--exclude':
            if len(dict_object[key]) == 0:
                continue
            for item in dict_object[key]:
                arg_list.append(key)
                arg_list.append(str(item))
        else:
            arg_list.append(key)
            arg_list.append(str(dict_object[key]))
        print(arg_list)

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='./example.jpeg', help='input source path. default example.jpeg')
    parser.add_argument('--output', type=str, default='./output', help='output source path, until filename(format exclude). default output')
    parser.add_argument('--option', type=str, default='skl', help='draw option. contains s, k, l skeleton, keypoint, label respectively. default skl')
    parser.add_argument('--exclude', nargs='+', type=int, default=-1, help='points to exclude. args for spacing. -1 for none(default), 0~17 to exclude. else error')
    parser.add_argument('--proto', type=str, default='./pose/coco/pose_deploy_linevec.prototxt', help='for model. default pose/coco/pose_deploy_linevec.prototxt')
    parser.add_argument('--weight', type=str, default='./pose/coco/pose_iter_440000.caffemodel', help='for model. default pose/coco/pose_iter_440000.caffemodel')
    parser.add_argument('--thres', type=float, default=0.1, help='set threshold for detecting. default 0.1')
    parser.add_argument('--gray', type=str2bool, default=False, help='preprocessing using gray img, set True')
    parser.add_argument('--back', type=str2bool, default=False, help='preprocessing removing background img, set True')
    parser.add_argument('--selectRect', type=str2bool, default=False, help='preprocessing select Rect to masking removed background img')
    parser.add_argument('--autolocation', type=str2bool, default=False, help='preprocessing using auto location to check boundary, set True')
    parser.add_argument('--comp', type=int, default=1, help='reducing fps only for video. fps/comp. default 1')
    parser.add_argument('--gamma', type=float, default=-1, help='gamma over 0. default -1(none)')
    parser.add_argument('--b_propo', type=str2bool, default=False, help='black proportion')

    opt = parser.parse_args(arg_list)
    print(opt)

    if fileformat(opt.source) == 0:
        forImage(opt)
    elif fileformat(opt.source) == 1:
        forVideo(opt)
    else:
        print('source file error')
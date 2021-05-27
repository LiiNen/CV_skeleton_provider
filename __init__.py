import argparse
import cv2
import time
import imutils
import math
import numpy as np
import sys,os

from CV_skeleton_provider.spcase.SPimage import forImage
from CV_skeleton_provider.spcase.SPvideo import forVideo
from CV_skeleton_provider.utils.formatter import str2bool
from CV_skeleton_provider.utils.formatter import fileformat


from CV_skeleton_provider.utils.formatter import optionChecker
from CV_skeleton_provider.utils.preprocessor import preBack
from CV_skeleton_provider.utils.preprocessor import preGray
from CV_skeleton_provider.utils.preprocessor import preGamma
from CV_skeleton_provider.utils.preprocessor import preBlackProportion
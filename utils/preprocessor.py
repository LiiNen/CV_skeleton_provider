import cv2
import numpy as np

def preBack(frame, selectRect_bool):
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if selectRect_bool:
        rect = cv2.selectROI(frame)
    else:
        rect = (10, 10, frame.shape[1] - 10, frame.shape[0] - 10)
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    frame = frame * mask2[:, :, np.newaxis]
    return frame

def preGray(frame, source):
    frame = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    bgr = frame[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # alpha = rgb2gray(frame)  # Channel 3
    frame = np.dstack([bgr])  # Add the alpha channel
    return frame
import cv2
import numpy as np

def preBack(frame, selectRect_bool, preBack_rect):
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if selectRect_bool:
        if preBack_rect == (0, 0, 0, 0):
            rect = cv2.selectROI(frame)
            preBack_rect = rect
        else:
            rect = preBack_rect
    else:
        rect = (10, 10, frame.shape[1] - 10, frame.shape[0] - 10)
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    frame = frame * mask2[:, :, np.newaxis]
    return frame, rect

def preGray(frame, source):
    frame = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    bgr = frame[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # alpha = rgb2gray(frame)  # Channel 3
    frame = np.dstack([bgr])  # Add the alpha channel
    return frame

def preGamma(frame, gamma):
    out = frame.copy()
    out = frame.astype(np.float)
    out = ((out / 255) ** (1 / gamma)) * 255
    out = out.astype(np.uint8)
    return out

def findWhite(frame):
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    output = np.array(output)
    h, w = output.shape[:2]
    colours, counts = np.unique(
        output.reshape(-1, 3), axis=0, return_counts=1)

    for index, colour in enumerate(colours):
        if(colour[0] == 0 & colour[1] == 0 & colour[2] == 0):
            cv2.imwrite('color.jpg', output)
            return (100*counts[index])/(h*w)

def preBlackProportion(frame):
    if findWhite(frame) < 30:
        image_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)  # YUV로 변경합니다.
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])  # 히스토그램 평활화를 적용
        image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        return image_rgb
    else:
        print('non needed')
        return frame

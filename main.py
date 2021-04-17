import cv2
import numpy as np
import os
import glob
from wshed import watershed_segmentation
from collections import deque
from skimage.segmentation import watershed
from copy import deepcopy

dir_path = "./data/task1-data-part"
filenames = glob.glob(os.path.join(dir_path + "2", "*.ts"))  # temporary path


cap = cv2.VideoCapture(filenames[20])
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_n = 0
kernel_open = np.ones((7,7),np.uint8)  # needs to be tuned
kernel_close = np.ones((33, 33), np.uint8)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0,0), fx=1., fy=1.)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if frame_n < 1:  # create for the first frame only
        canvas = np.zeros(gray.shape).astype(np.uint8)
    elif frame_n < 500:  # collect mask for the first 1000 frames
        fgmask = fgbg.apply(gray)  # get only moving pixels mask
        canvas = (canvas + fgmask).astype(np.uint8)
    canvas[canvas > 0] = 255
    opening = cv2.morphologyEx(255 - canvas, cv2.MORPH_OPEN, kernel_open, iterations=2)  # opening
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)  # closing
    res = cv2.bitwise_and(frame, frame, mask=(255 - closing))
    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    wshed = watershed_segmentation(res)
    res[wshed == 0] = (255, 0, 0)
    cv2.imshow('frame', res)
    frame_n += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()














# cap = cv2.VideoCapture(filenames[20])
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# frame_n = 0
# kernel_open = np.ones((7,7),np.uint8)  # needs to be tuned
# kernel_close = np.ones((33, 33), np.uint8)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.resize(frame, (0,0), fx=.8, fy=.8)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     if frame_n < 1:  # create for the first frame only
#         canvas = np.zeros(gray.shape).astype(np.uint8)
#     elif frame_n < 500:  # collect mask for the first 1000 frames
#         fgmask = fgbg.apply(gray)  # get only moving pixels mask
#         canvas = (canvas + fgmask).astype(np.uint8)
#     canvas[canvas > 0] = 255
#     opening = cv2.morphologyEx(255 - canvas, cv2.MORPH_OPEN, kernel_open)  # opening
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)  # closing
#     res = cv2.bitwise_and(frame, frame, mask=closing)
#     gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray_res)
#     frame_n += 1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()



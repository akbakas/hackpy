import cv2
import numpy as np
import os
import glob
from wshed import watershed_segmentation
from collections import deque
from funcutils import *

dir_path = "./data/task1-data-part"
filenames = glob.glob(os.path.join(dir_path + "2", "*.ts"))  # temporary path
cap = cv2.VideoCapture(filenames[20])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (800, 600), True)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_n = 0
kernel_open = np.ones((7,7),np.uint8)  # needs to be tuned
kernel_close = np.ones((33, 33), np.uint8)


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=30,
                      blockSize=7)

# Parameters for lucas
# kanade optical flow
lk_params = dict(winSize=(30, 30),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
counter = 1
speed = deque(maxlen=24)
stability = deque(maxlen=100)

shown_speed = 0
shown_stability = 0
ent_old = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0,0), fx=1, fy=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)



    if frame_n < 1:  # create for the first frame only
        canvas = np.zeros(gray.shape).astype(np.uint8)
    elif frame_n < 600:  # collect mask for the first 1000 frames
        fgmask = fgbg.apply(gray)  # get only moving pixels mask
        canvas = (canvas + fgmask).astype(np.uint8)
    canvas[canvas > 0] = 255
    opening = cv2.morphologyEx(255 - canvas, cv2.MORPH_OPEN, kernel_open, iterations=2)  # opening
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)  # closing
    res = cv2.bitwise_and(frame, frame, mask=(255 - closing))
    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # without watershed algo
    wshed = watershed_segmentation(res)
    res[wshed == 0] = (255, 0, 0)
    cv2.imshow('frame', gray_res)
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



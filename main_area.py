import cv2
import numpy as np
import os
import glob
from wshed import watershed_segmentation
from collections import deque
from blobdet import count_big_bubbles
import json
from argparse import ArgumentParser

dir_path = "./data/task1-data-part1"
# filenames = glob.glob(os.path.join(dir_path + "2", "*.ts"))  # temporary path
filenames = ['F1_1_3_1.ts', 'F1_1_3_2.ts', 'F1_1_4_1.ts', 'F1_1_4_2.ts', 'F1_2_3_1.ts', 'F1_2_3_2.ts', 'F2_1_2_2.ts']

parser = ArgumentParser()

parser.add_argument(
        '--video', required=True,
        help="Path to the video."
        )
args = parser.parse_args()

filename = os.path.expanduser(args.video)
print(f'file: {filename} started processing')

print(filename[:-3])
cap = cv2.VideoCapture(filename)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(f"{filename[:-3]}" + ".mp4", fourcc, 30.0, (400, 600))

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_n = 0
kernel_open = np.ones((7,7),np.uint8)  # needs to be tuned
kernel_close = np.ones((33, 33), np.uint8)

queue = deque(maxlen=30)

i = 0  # dummy counter
row = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('No ret detected!')
        break

    frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
    frame = frame[:400, :, :]
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
    wshed = watershed_segmentation(frame)
    frame[wshed == 0] = (255, 0, 0)
    res = cv2.bitwise_and(frame, frame, mask=(255 - closing))
    gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # without watershed algo
    big_bubble_num = count_big_bubbles(frame)  # the number of big bubbles
    wshed = cv2.bitwise_and(wshed.astype(np.uint8), wshed.astype(np.uint8), mask=(255-closing))
    values, areas = np.unique(wshed, return_counts=True)
    queue.appendleft(np.mean(areas))

    # if frame_n % 150 == 0:
    #     print(np.mean(queue))
    #     for_json = {'area': np.mean(queue)}
    #     row.append(for_json)
    #     with open(f"{filename[:-3]}" + ".json", 'w') as outfile:
    #         json.dump(row, outfile)

    cv2.imshow('frame', res)
    # out.write(res)
    frame_n += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1
cv2.destroyAllWindows()
cap.release()





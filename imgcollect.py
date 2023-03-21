import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

import uuid
import os
import time


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


IMAGES_PATH = os.path.join('data', 'images')
labels = ['cone']
number_imgs = 30

cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for i in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, i))

        ret, frame = cap.read()

        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('Image Collection', frame)
        time.sleep(2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

import uuid
import os
import time


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt', force_reload=True)


cap = cv2.VideoCapture(0)
while(cap.isOpened()):

    ret, frame = cap.read()

    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

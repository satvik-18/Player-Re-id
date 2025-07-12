import cv2
from typing import List, Dict
from ultralytics import YOLO
import numpy as np

#Define a function to detect players and ball for a single image
model = YOLO(r"models\best.pt")
def yolodetection(img, conf_threshold = 0.7):

    results = model(img, verbose = False, iou = 0.7)[0]
    detections = []
    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        bbox = [x1,y1,x2-x1,y2-y1]
        score = float(box.conf[0])
        ids = int(box.cls[0])
        label = model.names[ids]
        if(score>conf_threshold and label == 'player'):
            detections.append((bbox, score, ids))
    return detections
        
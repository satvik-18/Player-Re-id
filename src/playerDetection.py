import cv2
from typing import List, Dict
from ultralytics import YOLO
import numpy as np

#Define a function to detect players and ball

def yolodetection(video_path: str, modelpath: str, save_output: bool = True )->List[List[Dict]]:
    model = YOLO(modelpath)

    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if(save_output == True):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/yolo_output.mp4', fourcc,fps, (w,h))

    detected = []

    while True:
        Success, img = cap.read()

        if not Success or img is None or img.size == 0:
            break

        results = model(img, verbose = False )[0]
        frameDetections = []

        for  box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            score = np.ceil(box.conf[0]*100)/100

            ids = int(box.cls.item())

            label = model.names[ids]
            print(f"[✅] Detected {label} with confidence {score}")

            frameDetections.append({
                "bbox": [x1,y1, x2, y2],
                "conf": score,
                "class_id" : ids,
                "label": label
            })
            if(score > 0.4):
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, f"{label} {score:0.2f}", (max(x1,0),max(y1,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        detected.append(frameDetections)

        if save_output: 
            out.write(img)

        cv2.imshow("op", img)
        if(cv2.waitKey(200) and 0xFF == ord('q')):
            break
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()
    return detected

yolodetection(video_path=r"assets\15sec_input_720p.mp4", modelpath=r"models\best.pt")
import numpy as np

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

from playerDetection import yolodetection
import cv2
from ultralytics import YOLO
from embeddingExtractor import extractPlayerEmbeddings
from utils import crop, resize
from collections import deque

# Update tracker with current frame detections.
model = YOLO(r"models\best.pt")


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def tracking(video_path):

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, 100)
    tracker = Tracker(metric, max_age= 60, n_init=5 ,max_iou_distance=0.5)

    opPath = r'output\yolo_output_with_embeddings.mp4'
    
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(opPath, fourcc, fps, (frame_width, frame_height))
    
    while True:

        Success, img = cap.read()
        if not Success:
            break
        detections = yolodetection(img, conf_threshold=0.6)
        
        enriched_detections =[]

        MIN_WIDTH = 10
        MIN_HEIGHT = 20
    
        for (bbox, score, class_id_yolo) in detections:
            x1,y1,w,h = bbox

            if(w<MIN_WIDTH or h<MIN_HEIGHT):
                continue

            x2,y2 = x1+w, y1+h

            crop_img = crop(img, x1,y1,x2,y2, pad =20)
            crop_img = resize(crop_img)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            embedding = extractPlayerEmbeddings(crop_img)
            if embedding is None or np.linalg.norm(embedding) < 1e-3:
                continue
            d = Detection([x1,y1,w,h], score,embedding)
            d.class_id = class_id_yolo
            enriched_detections.append(d)
        
        enriched_detections.sort(key=lambda d: d.confidence, reverse=True)
        filtered_detections = []
        for det in enriched_detections:
            overlaps = [compute_iou(det.tlwh, d.tlwh) for d in filtered_detections]
            if all(iou < 0.6 for iou in overlaps):
                filtered_detections.append(det)
        enriched_detections = filtered_detections
        print(f"Raw detections: {len(detections)}, after embedding filter: {len(enriched_detections) + len(filtered_detections)}, after overlap filter: {len(filtered_detections)}")



        tracker.predict()
        tracker.update(enriched_detections)

        for tr in tracker.tracks:
            print(f"Track {tr.track_id} | Confirmed: {tr.is_confirmed()} | Missed frames: {tr.time_since_update}")

            if not tr.is_confirmed() or tr.time_since_update > 1:
                continue

            best_det = None
            best_iou = 0.0

            for d in enriched_detections:
                iou = compute_iou(d.tlwh, tr.to_tlwh())
                if iou>best_iou and iou>0.5:
                    best_det = d
                    best_iou = iou

            if best_det:
                if not hasattr(tr, 'features'):
                    tr.features = deque(maxlen = 20)
                tr.features.append(best_det.feature)
                tr.averaged_embedding = np.mean(tr.features,axis = 0)
            label = "Player"

            l, t, w, h = map(int,tr.to_tlwh())
            tid = tr.track_id
            cv2.rectangle(img, (l,t), (l+w, t+h), (0,0,255), 2)
            cv2.putText(img, f"{label}#{tid}", (max(l,0),max(t,0)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),1)
        out.write(img)
        cv2.imshow("Tracking", img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


tracking(r"assets\15sec_input_720p.mp4")
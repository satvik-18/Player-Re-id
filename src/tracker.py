import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from playerDetection import yolodetection
import cv2
from ultralytics import YOLO
from embeddingExtractor import extractPlayerEmbeddings

# Update tracker with current frame detections.
model = r"models\best.pt"
def tracking(video_path):

    tracker = DeepSort(max_age=40, n_init=3, nms_max_overlap=1.0,max_cosine_distance=0.2, half = False, bgr = True, embedder_gpu=False)
    opPath = 'output\yolo_output_with_embeddings.mp4'
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
        detections = yolodetection(img, conf_threshold=0.7)
        enriched_detections = []

        for (bbox, score, label) in detections:
            x1,y1,w,h = bbox
            x2,y2 = x1+w, y1+h

            crop = img[y1:y2, x1:x2]
            embedding = extractPlayerEmbeddings(crop)
            if embedding is None:
                continue

            enriched_detections.append(([x1,y1,w,h], float(score), label, embedding))

        tracked = tracker.update_tracks(enriched_detections, frame = img)

        for tr in tracked:
            
            if not tr.is_confirmed():
                continue

            l,t,r,b = map(int, tr.to_ltrb())
            tid = tr.track_id
            
            label = tr.get_det_class() or "Unknown"

            cv2.rectangle(img, (l,t), (r,b), (0,0,255), 2)
            cv2.putText(img, f"{label}#{tid}", (max(l,0),max(t,0)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),1)
        out.write(img)
        cv2.imshow("Tracking", img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
tracking(r"assets\15sec_input_720p.mp4")
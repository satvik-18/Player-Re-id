import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from playerDetection import yolodetection
import cv2
from ultralytics import YOLO
from embeddingExtractor import extractPlayerEmbeddings
from utils import crop, resize

# Update tracker with current frame detections.
model = YOLO(r"models\best.pt")
def tracking(video_path):

    tracker = DeepSort(max_age=15, n_init=3, nms_max_overlap=0.5,max_cosine_distance=0.4, half = False, bgr = True, embedder_gpu=False)
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
        detections = yolodetection(img, conf_threshold=0.5)
        
        filtered_detections = []
        embeddings = []

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

            if embedding is None:
                print(f"[WARNING] No embedding extracted for bbox: {bbox}")
                continue
            elif np.linalg.norm(embedding) < 1e-3:
                print(f"[WARNING] Extracted embedding too small (nearly zero norm) for bbox: {bbox}")
                continue
            else:
                print(f"[INFO] Valid embedding extracted for bbox: {bbox} | Norm: {np.linalg.norm(embedding):.4f}")


            filtered_detections.append((bbox, score, class_id_yolo))
            embeddings.append( embedding )

        print(f"[DeepSort INPUT] Count: {len(filtered_detections)} | Embeddings: {len(embeddings)}")

        tracked = tracker.update_tracks(filtered_detections, embeds= embeddings, frame = img)
        print(f"[FRAME] Tracked boxes: {sum(tr.is_confirmed() for tr in tracked)}") 
        for i,tr in enumerate(tracked):
            print(f"[TRACKER] Confirmed: {tr.is_confirmed()} | Track ID: {tr.track_id}")

            if not tr.is_confirmed():
                continue

            l,t,r,b = map(int, tr.to_ltrb())
            tid = tr.track_id

            current_embedding = None
            if hasattr(tr, 'det_index') and tr.det_index is not None:
                det_idx = tr.det_index
                if 0 <= det_idx < len(embeddings):
                    current_embedding = embeddings[det_idx]

            if current_embedding is not None:
                if hasattr(tr, 'features'):
                    tr.features.append(current_embedding)
                    if len(tr.features) > 10:
                        tr.features.pop(0)
                else:
                    tr.features = [current_embedding]

                tr.averaged_embedding = np.mean(tr.features, axis=0)
                print(f"[TRACK #{tid}] Averaged embedding norm: {np.linalg.norm(tr.averaged_embedding):.4f} | History length: {len(tr.features)}")

            c_id = tr.get_det_class()
            if(c_id == None):
                label = 'Unknown'
            else:
                label = model.names[c_id]

            if hasattr(tr, 'mean'):
                mean = tr.mean
                cx, cy, a, h = mean[0], mean[1], mean[2], mean[3]

                w = a * h
                pl = int(cx - w / 2)
                pr = int(cx + w / 2)
                pt = int(cy - h / 2)
                pb = int(cy + h / 2)
                cv2.rectangle(img, (pl, pt), (pr, pb), (255, 200, 100), 1) 
            
            cv2.rectangle(img, (l,t), (r,b), (0,0,255), 2)
            cv2.putText(img, f"{label}#{tid}", (max(l,0),max(t,0)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),1)
        out.write(img)
        cv2.imshow("Tracking", img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


tracking(r"assets\15sec_input_720p.mp4")
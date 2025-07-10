import torch
from torchreid.reid.utils import FeatureExtractor
import numpy as np
import cv2

extractor = FeatureExtractor(model_name='osnet_x0_25', model_path='', device = 'cpu')

def extractPlayerEmbeddings(crop_img):
    try:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        features = extractor([crop_rgb])
        return features[0].cpu().numpy()
    
    except Exception as e:
        print(f"[ERROR] Re-ID embedding failed : {e}")
        return None

print("Done")


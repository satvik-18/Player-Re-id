import torch
from torchreid.utils import FeatureExtractor
import numpy as np
import cv2

extractor = FeatureExtractor(model_name='osnet_ain_x1_0', model_path='', device = 'cpu')

def extractPlayerEmbeddings(crop_img):
    try:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        features = extractor([crop_rgb])
        
        if isinstance(features, torch.Tensor):
            features = features.squeeze(0).cpu().numpy()

        else:
            raise ValueError("Unexpected feature shape/type")

        features = features / (np.linalg.norm(features) + 1e-6)  # Normalize
        return features
    
    except Exception as e:
        print(f"[ERROR] Re-ID embedding failed : {e}")
        return None



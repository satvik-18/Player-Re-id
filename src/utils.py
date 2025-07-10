import cv2
import numpy as np

def pad_resize(crop, size = (256, 128)):

    h,w = crop.shape[:2]

    if(h==0 or w == 0):
        return None
    
    scale = min(size[0]/h, size[1]/w)
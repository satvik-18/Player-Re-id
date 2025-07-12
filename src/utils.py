import cv2
import numpy as np

def crop(img, x1, y1, x2, y2, pad=0):

    if img is None or img.size == 0:
        print("Error: Input image to crop is empty or None. Returning black 1x1 image.")
        return np.zeros((1, 1, 3), dtype=np.uint8) 

    h, w = img.shape[:2]

    
    x1_padded = max(0, x1 - pad)
    y1_padded = max(0, y1 - pad)
    x2_padded = min(w, x2 + pad)
    y2_padded = min(h, y2 + pad)

    x1_final = int(x1_padded)
    y1_final = int(y1_padded)
    x2_final = int(x2_padded)
    y2_final = int(y2_padded)

    if x1_final >= x2_final or y1_final >= y2_final:
        print(f"Warning: Invalid crop region after padding. x1={x1_final}, y1={y1_final}, x2={x2_final}, y2={y2_final}. Returning black 1x1 image.")
        
        return np.zeros((1, 1, 3), dtype=np.uint8) 

    try:
        cropped_img = img[y1_final:y2_final, x1_final:x2_final]

        if cropped_img.size == 0:
            print(f"Warning: Cropped image has zero size after slicing. x1={x1_final}, y1={y1_final}, x2={x2_final}, y2={y2_final}. Returning black 1x1 image.")
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return cropped_img
    except Exception as e:
        print(f"Error during image cropping: {e}. Returning black 1x1 image.")
        return np.zeros((1, 1, 3), dtype=np.uint8)

def resize(img, target_size=(64, 128)):
    if img is None or img.size == 0:
        print("Error: Input image to resize is empty or None. Returning black image of target size.")
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8) 

    try:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        if resized_img.size == 0:
            print(f"Warning: Resized image has zero size. Returning black image of target size.")
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return resized_img
    except Exception as e:
        print(f"Error during image resizing: {e}. Returning black image of target size.")
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8) 

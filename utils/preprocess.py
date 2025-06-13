import cv2
import numpy as np

def preprocess_image(img, size=(32, 32)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    img_normalized = img_enhanced.astype(np.float32) / 255.0
    
    return img_normalized

def preprocess_image_advanced(img, size=(32, 32)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_denoised = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    img_resized = cv2.resize(img_denoised, size, interpolation=cv2.INTER_AREA)
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    img_lab[:,:,0] = cv2.equalizeHist(img_lab[:,:,0])
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    img_normalized = img_enhanced.astype(np.float32) / 255.0
    return img_normalized
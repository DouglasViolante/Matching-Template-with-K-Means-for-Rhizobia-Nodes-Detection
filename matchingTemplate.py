import cv2
import numpy as np

def template_Matching(img_rgb, template_image):

    img_rgb = cv2.medianBlur(img_rgb, 7)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    
    res = cv2.matchTemplate(img_gray, template_image, cv2.TM_CCOEFF_NORMED)

    detection_threshold = 0.6

    match_locations = np.where(res >= detection_threshold)

    return match_locations

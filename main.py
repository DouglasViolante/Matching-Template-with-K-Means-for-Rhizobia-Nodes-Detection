import cv2
import numpy as np

import sys

np.set_printoptions(threshold = sys.maxsize)

import matchingTemplate as mT
import matchingLocationsKMeans as mLKM


img_rgb = cv2.imread('Exp.1 - Florestópolis, 35DAE, 2017-2018, T15, R1, P2.jpg')

template_image = cv2.imread('TemplateNode.jpg', 0)
template_image = cv2.medianBlur(template_image, 5)
width, height  = template_image.shape[::-1]


match_locations = mT.template_Matching(img_rgb, template_image)
labels_array =     mLKM.matched_locations_KMeans(match_locations)

SIZE_ARRAY_MATCH_LOCATIONS = len(list(zip(*match_locations[::-1])))

i = 0

for position_tuple in zip(*match_locations[::-1]):
    
    if(i < SIZE_ARRAY_MATCH_LOCATIONS):
        if(labels_array[i] == 1):
            cv2.rectangle(img_rgb, position_tuple, (position_tuple[0] + width, position_tuple[1] + height), (0,255,0), 1)
        else:
            cv2.rectangle(img_rgb, position_tuple, (position_tuple[0] + width, position_tuple[1] + height), (0,0,255), 1)
    
    i = i + 1


cv2.namedWindow('Classified Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Classified Output', 800,600)
cv2.imshow('Classified Output', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
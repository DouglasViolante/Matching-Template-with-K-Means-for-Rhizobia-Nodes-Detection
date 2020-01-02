import numpy as np
import cv2

RADIUS_CONST = 1

def matched_locations_KMeans(match_locations):

    match_locations = np.float32((np.array(match_locations).T))

    # define criteria and apply kmeans()
    criteria     = (cv2.TERM_CRITERIA_EPS, 100, 1.0)
    _, labels, _ = cv2.kmeans(match_locations, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return labels

    


    

    


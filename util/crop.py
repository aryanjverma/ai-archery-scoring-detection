import cv2
from geometry import *
import numpy as np

def crop(contours, image):
    sourcePoints = findLargestSquare(contours)
    sourcePoints = orderPoints(sourcePoints)
    W, H = 1000, 1000
    destinationPoints = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]
    ], dtype="float32")

    T = cv2.getPerspectiveTransform(sourcePoints, destinationPoints)
    warped = cv2.warpPerspective(image, T, (W, H))
    return warped
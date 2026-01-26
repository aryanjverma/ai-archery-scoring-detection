from targetTypes import TARGET_SPECS
import cv2
from geometry import findCenters, findCircles, findRadii

def identifyTarget(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    centers = findCenters(gray)
    print(centers)
    if len(centers) == 5:
        return TARGET_SPECS[1]
    elif len(centers) == 3:
        return TARGET_SPECS[2]
    else:
        edges = cv2.Canny(
            gray,
            threshold1=80,
            threshold2=160
        )
        edges = cv2.GaussianBlur(edges, (3,3), 0)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        circles = findCircles(contours, 100)
        radiiMap = findRadii(centers, circles, min(gray.shape[0], gray.shape[1]))
        for center in radiiMap.keys():
            if len(radiiMap[center]) > 6:
                return TARGET_SPECS[0]
            else:
                return TARGET_SPECS[3]
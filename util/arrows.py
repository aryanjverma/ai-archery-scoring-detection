import cv2
import numpy as np

def findArrowPosition(before, after):
    image = cv2.absdiff(before, after)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(grayImage, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('diff.png', thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrowImage = np.zeros((1000, 1000), dtype=np.uint8)
    # finds largest area object and makes it nock
    nockContour = None
    nockArea = 0
    for index in range(len(contours)):
        contour = contours[index]
        area = cv2.contourArea(contour)
        print(area)
        if area > nockArea:
            nockContour = contour
            nockArea = area
    M = cv2.moments(nockContour)
    nockCenterX = int(M["m10"] / M["m00"])
    nockCenterY = int(M["m01"] / M["m00"])
    
    areaCutoff = 5
    distanceCutoff = 100
    for index in range(len(contours)):
        contour = contours[index]
        area = cv2.contourArea(contour)
        if area > areaCutoff:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            distance = np.sqrt((nockCenterX - cX) ** 2 + (nockCenterY - cY) ** 2)
            if distance < distanceCutoff:
                cv2.drawContours(arrowImage, [contour], 0, (255,255,255), -1)
    pts = np.column_stack(np.where(arrowImage > 0))[:, ::-1]
    cv2.imwrite('diff.png', arrowImage)
    return pts[pts[:, 0].argmax()]
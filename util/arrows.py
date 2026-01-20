import cv2
import numpy as np

def findArrowPosition(before, after):
    
    image = cv2.absdiff(before, after)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, contours, -1, (0,255,0), 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = np.column_stack(np.where(thresh > 0))[:, ::-1]
    
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis /= np.linalg.norm(axis)

    projections = (pts - mean) @ axis
    offsetDistance = 15
    end1 = pts[np.argmin(projections)] + offsetDistance * axis
    end2 = pts[np.argmax(projections)] - offsetDistance * axis
    
    
    thickness1 = localThickness(pts, end1)
    thickness2 = localThickness(pts, end2)
    
    impactPoint = 0
    
    if thickness1 < thickness2:
        impactPoint = end1
    else:
        impactPoint = end2
    cv2.circle(after, (int(impactPoint[0]), int(impactPoint[1])), 20, (0,255,0),1)
    return impactPoint
def localThickness(pts, endpoint, radius=6):
    dists = np.linalg.norm(pts - endpoint, axis=1)
    return np.sum(dists < radius)

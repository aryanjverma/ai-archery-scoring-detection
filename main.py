import cv2
import math

image = cv2.imread('reallife.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(
    gray,
    threshold1=80,
    threshold2=160
)
edges = cv2.GaussianBlur(edges, (5,5), 0)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sizeCutoff = 50
def findDeepest(hierarchy, contours):
    checked = [False] * len(hierarchy[0])
    deepestDepth = 0
    deepestIndex = 0
    for index in range(len(hierarchy[0])):
        if not checked[index]:
            current = index
            while not checked[current] and hierarchy[0][current][2] != -1:
                checked[current] = True
                current = hierarchy[0][current][2]
            if checked[current]:
                break
            depth = 0
            lowest = current
            while hierarchy[0][current][3] != -1:
                hierarchy[0][current][0] = -2
                depth += 1
                current = hierarchy[0][current][3]
            area = cv2.contourArea(contours[lowest], True)
            if depth > deepestDepth and area > 50:
                deepestDepth = depth
                deepestIndex = lowest
    return deepestIndex
def findCenter(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        return (cx, cy)
deepestIndex = findDeepest(hierarchy, contours)
cx, cy = findCenter(contours[deepestIndex])

circularityCutoff = 0.8
areaMin = 3000
areaMax = 700000
circles = []

for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter != 0:
        circularity = (4 * math.pi * area) / (perimeter**2)
        if circularity > circularityCutoff and areaMin < area < areaMax:
            circles.append(cv2.minEnclosingCircle(contour))
radiiSums = []
radiiCounts = []
distanceCutoff = 20
radiusCutoff = 20
for circle in circles:
    (x, y), r = circle
    if abs(cx - x) < distanceCutoff and abs(cy - y) < distanceCutoff:
        needToAdd = True
        for index in range(len(radiiSums)):
            if abs(radiiSums[index] - r) < radiusCutoff:
                radiiSums[index] += r
                radiiCounts[index] += 1
                needToAdd = False
                break
        if needToAdd:
            radiiSums.append(r)
            radiiCounts.append(1)
radii = []
for index in range(len(radiiCounts)):
    radii.append(radiiSums[index] / radiiCounts[index])
for radius in radii:
    cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
print(radii)
cv2.imshow('Contours', image)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


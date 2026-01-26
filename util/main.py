import cv2
from target import Target
from identifyTarget import identifyTarget
from crop import crop
start = cv2.imread('5before.png')
start = crop(start)
targetType = identifyTarget(start)
print(targetType.name)
exit(3)
target = Target()
start = cv2.imread('5before.png')
next = cv2.imread('5after.png')

target.updateScore(start, next)
print(target.score)

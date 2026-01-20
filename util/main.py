import cv2
from target import Target
from targetTypes import TARGET_SPECS

targetName = "NFAA 1 Spot"
targetSpec = next(spec for spec in TARGET_SPECS if spec.name == targetName)
target = Target(targetSpec)
start = cv2.imread('2arrow.png')
next = cv2.imread('3arrow.png')
target.updateScore(start, next)
print(target.score)
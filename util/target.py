from arrows import findArrowPosition
import math
import numpy as np
import functools
from crop import crop
import cv2

@functools.total_ordering
class Target:
    def __init__(self, targetType):
        self.targetType = targetType
        self.centers = targetType.centers
        self.radii = targetType.radii
        self.numCircles = len(self.radii)
        self.score = 0
        self.xCount = 0
        
    def __eq__(self, other):
        if not isinstance(other, Target):
            raise TypeError('Comparing target to non-target is illegal')
        if self.targetType != other.targetType:
            raise TypeError('Comparing differnt target types is illegal')
        return self.score == other.score and self.xCount == other.xCount
    def __lt__(self, other):
        if not isinstance(other, Target):
            raise TypeError('Comparing target to non-target is illegal')
        if self.targetType != other.targetType:
            raise TypeError('Comparing differnt target types is illegal')
        return self.score < other.score or (self.score == other.score and self.xCount < other.xCount)

    def updateScore(self, before, after):
        before = crop(before)
        after = crop(after)
        newArrowPosition = findArrowPosition(before, after)
        circlePositions = np.zeros(shape=(self.numCircles+1))
        circlePositions[self.locateInnerCirclePosition(newArrowPosition, after)] += 1
        
        self.score += np.dot(circlePositions, self.targetType.scoreVector)
        self.xCount += circlePositions[0]
        cv2.imwrite('diff.png', after)
    def locateInnerCirclePosition(self, position, after):
        approxArrowRadius = 25
        for center in self.centers:
            distance = math.sqrt((position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2)
            distance -= approxArrowRadius
            if self.radii[self.numCircles - 1] > distance:
                count = 0
                while (count < len(self.radii) and self.radii[count] < distance):
                    count += 1
                print(count)
                return count
        return self.numCircles
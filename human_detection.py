import cv2
import numpy as np

def findHumanContours(frame, background_subtractor):
    fgMask = background_subtractor.apply(frame, frame, 0.2)
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

    contours, _hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda c: cv2.contourArea(c) > 1000, contours))

    return contours, fgMask

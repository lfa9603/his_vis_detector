import cv2
import numpy as np
from assignment.Contours import Contours
from assignment.utils.wait_for_photo import wait_for_photo

def findHumanContours(frame, background_subtractor):
    fgMask = background_subtractor.apply(frame, frame, 0.2)
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

    contours, _hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda c: cv2.contourArea(c) > 1000, contours))
    # Contours.drawContours(contours=contours, frame=frame)
    # wait_for_photo(closing)
    # wait_for_photo(frame)

    # cv2.imshow('dfdf', np.hstack([frame]))

    return contours, fgMask






# if __name__ == '__main__':
#     capture = cv2.VideoCapture(0)
#     if not capture.isOpened():
#         print('Unable to capture vidio')
#         exit(0)
#
#     fgbg = cv2.createBackgroundSubtractorMOG2()
#     while True:
#         ret, frame = capture.read()
#         if frame is None:
#             print('Unable to read frame')
#             exit(0)
#
#         ## [apply]
#         # update the background model
#         fgMask = fgbg.apply(frame, frame, 0.15)
#         kernel = np.ones((15, 15), np.uint8)
#         closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
#
#         contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         Contours.drawContours(contours=contours, frame=frame)
#
#         cv2.imshow("HSV Frane and non-foreground-masked background", np.hstack([frame]))
#
#         ## [show]
#
#         keyboard = cv2.waitKey(30)
#         if keyboard == 'q' or keyboard == 27:
#             break

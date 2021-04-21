
import numpy as np
import cv2
from assignment.Contours import Contours
from assignment.binarize import binarizeFrame
from assignment.human_detection import findHumanContours



capture = cv2.VideoCapture('./output1.avi')
if not capture.isOpened():
    print('Unable to open: ;;')
    exit(0)

BACKGROUND_SUBTRACTOR_MOG2 = cv2.createBackgroundSubtractorMOG2()  # instantiate background subtractor MOG2
COLOR_BOUNDARIES = [  # define the list of boundaries for colors to pick
        # orange hsv
        ([0, 160, 85], [12, 255, 255]),
        # yellow hsv
        ([31, 150, 50], [36, 255, 255]),
    ]


while True:
    ret, frame = capture.read()
    if frame is None:
        print('No frame')
        break

    human_contours, human_contour_fg_mask = findHumanContours(frame, background_subtractor=BACKGROUND_SUBTRACTOR_MOG2)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.GaussianBlur(hsv_frame, (19, 19), 0) # Gaussian blur to reduce noise in the image.

    output = None
    # pass the frame to the background subtractor
    foreground_mask = BACKGROUND_SUBTRACTOR_MOG2.apply(hsv_frame)

    for (lower, upper) in COLOR_BOUNDARIES:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # the mask
        mask = cv2.inRange(hsv_frame, lower, upper)
        color_output = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

        if output is None:
            output = color_output
        else:
            output = cv2.bitwise_or(output, color_output)

    # trying to remove white straps by dilating spotted colors
    kernel = np.ones((13, 13), np.uint8)
    closing = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

    binarized = binarizeFrame(closing)

    # Looking for contours
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for vest_contour in contours:
        for human_contour in human_contours:
            if Contours.is_countour_within(vest_contour, human_contour):
                area_percentage = cv2.contourArea(vest_contour) / cv2.contourArea(human_contour)
                if (area_percentage > 0.1):
                    Contours.drawContours([vest_contour], frame=output)
                    Contours.drawContours([human_contour], frame=output, color=(0, 0, 255))

            else:
                if (cv2.contourArea(human_contour) > 20000):
                    Contours.drawContours([human_contour], frame=output, color=(255, 255, 0))


    # cv2.imshow('Masking the background', cv2.bitwise_and(output, output, mask=foreground_mask))
    cv2.imshow("HSV Frane and non-foreground-masked background", np.hstack([hsv_frame, output]))
    cv2.imshow("HSV Frane and non-foreground-masked sd", np.hstack([hsv_frame, output]))
    # cv2.imshow("HSV Frane and non-foreground-masked background", np.hstack([human_contour_fg_mask]))

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


import numpy as np
import cv2
from assignment.utils.Contours import Contours
from assignment.utils.binarize import binarizeFrame
from assignment.human_detection import findHumanContours


capture = cv2.VideoCapture('./official_videos/presentation_video.avi')


if not capture.isOpened():
    print('Unable to open capture')
    exit(0)

BACKGROUND_SUBTRACTOR_MOG2 = cv2.createBackgroundSubtractorMOG2()  # instantiate background subtractor MOG2
COLOR_BOUNDARIES = [  # define the list of boundaries for colors to pick
        # orange hsv
        ([0, 160, 85], [12, 255, 255]),
        # yellow hsv
        # ([31, 150, 50], [36, 255, 255]),
    ]

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

while True:
    ret, main_frame = capture.read()

    if main_frame is None:
        print('No main_frame')
        break

    human_contours, human_contour_fg_mask = findHumanContours(main_frame, background_subtractor=BACKGROUND_SUBTRACTOR_MOG2)

    hsv_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.GaussianBlur(hsv_frame, (19, 19), 0) # Gaussian blur to reduce noise in the image.

    output = None
    # pass the main_frame to the background subtractor
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

    for human_contour in human_contours:
        found_vest = False
        vest_pieces_for_human = []
        for vest_contour in contours:

            if Contours.is_countour_within(vest_contour, human_contour):
                vest_pieces_for_human.append(vest_contour)
                found_vest = True

        if found_vest:
            area_percentage = sum(cv2.contourArea(v_c) for v_c in vest_pieces_for_human) / cv2.contourArea(human_contour)

            if (area_percentage > 0.1):
                Contours.drawContours([human_contour], frame=output, color=(0, 255, 0))

        else:
            if (cv2.contourArea(human_contour) > 2000):
                Contours.drawContours([human_contour], frame=output, color=(0, 0, 255))


    cv2.putText(main_frame, 'Input', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(output, 'Output', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("HSV Frane and non-foreground-masked sd", np.hstack([main_frame, output]))

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


import numpy as np
import cv2


def drawContours(contours):
    for contour in contours:
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

        area = cv2.contourArea(contour)
        print(area)
        if area < 10000:  # Set a lower bound on the elipse area.
            continue

        cv2.drawContours(output, contour, -1, (0, 0, 255), 1)
        cv2.rectangle(output, (leftmost[0], topmost[1]), (rightmost[0], bottommost[1]), (0, 255, 0), 2)  # Draw the ellipse on the original image.


def binarizeFrame(f):
    # binarizing image
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (19, 19), 0)  # Gaussian blur to reduce noise in the image.
    # Use adaptive thresholding to "binarize" the image.
    binarized = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    return binarized



capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open: ;;')
    exit(0)

BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorKNN()  # instantiate background subtractor
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

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.GaussianBlur(hsv_frame, (19, 19), 0) # Gaussian blur to reduce noise in the image.


    output = None
    # pass the frame to the background subtractor
    foreground_mask = BACKGROUND_SUBTRACTOR.apply(hsv_frame)

    # contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=3)

    binarized = binarizeFrame(closing)

    # Looking for contours
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawContours(contours)

    # cv2.imshow('Masking the background', cv2.bitwise_and(output, output, mask=foreground_mask))
    cv2.imshow("HSV Frane and non-foreground-masked background", np.hstack([hsv_frame, output]))

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

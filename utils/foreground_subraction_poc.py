from __future__ import print_function
import cv2


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open: ;;')
    exit(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    #update the background model
    fgMask = fgbg.apply(frame, frame, 0.15)

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    #show the current frame and the fg masks
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
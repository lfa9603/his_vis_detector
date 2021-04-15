import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def nothing(x):
    pass
# Creating a window for later use
cv2.namedWindow('result')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h_min', 'result',0,179,nothing)
cv2.createTrackbar('s_min', 'result',0,255,nothing)
cv2.createTrackbar('v_min', 'result',0,255,nothing)
# Creating track bar
cv2.createTrackbar('h_max', 'result',0,179,nothing)
cv2.createTrackbar('s_max', 'result',0,255,nothing)
cv2.createTrackbar('v_max', 'result',0,255,nothing)

while(1):

    _, frame = cap.read()

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # get info from track bar and appy to result
    h_min = cv2.getTrackbarPos('h_min','result')
    s_min = cv2.getTrackbarPos('s_min','result')
    v_min = cv2.getTrackbarPos('v_min','result')

    h_max = cv2.getTrackbarPos('h_max', 'result')
    s_max = cv2.getTrackbarPos('s_max', 'result')
    v_max = cv2.getTrackbarPos('v_max', 'result')
    # Normal masking algorithm
    lower_blue = np.array([h_min,s_min,v_min])
    print('--------------------')
    print('MIN: hsv({}, {}, {})'.format(h_min, s_min, v_min))
    print('MAX: hsv({}, {}, {})'.format(h_max, s_max, v_max))
    print('--------------------')

    upper_blue = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv,lower_blue, upper_blue)

    result = cv2.bitwise_and(frame,frame,mask = mask)
    cv2.putText(result, 'MIN: hsv({}, {}, {})'.format(h_min, s_min, v_min), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(result, 'MAX: hsv({}, {}, {})'.format(h_max, s_max, v_max), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
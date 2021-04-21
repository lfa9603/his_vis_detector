
import cv2
import uuid

def wait_for_photo(frame):
    keyboard = cv2.waitKey(30)
    if keyboard == 112: # 'p' for photo
        cv2.imwrite("frame{}.jpg".format(str(uuid.uuid1())), frame)  # save frame as JPEG file
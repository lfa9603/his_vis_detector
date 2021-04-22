import cv2
import uuid
# Open the first camera connected to the computer.
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('not-correct-wear-{}.avi'.format(uuid.uuid1()), fourcc, 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # Read an frame from the webcam.

    out.write(frame)  # Write the frame to the output file.
    print('saving to file', frame)

    cv2.imshow('frame', frame)  # While we're here, we might as well show it on the screen.

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera device and output file, and close the GUI.
cap.release()
out.release()
cv2.destroyAllWindows()

import cv2

class Contours:


    @staticmethod
    def is_countour_within(child, parent):
        """
        --------------->
        |
        |
        |
        |
        V

        Type of system, checking contour must be done considering the
        negative y-axis (descending 0/-inf) as asc (0/+inf)
        :param child:
        :param parent:
        :return:
        """
        child_coordinates = Contours.find_contour_coordinates(child)
        parent_coordinates = Contours.find_contour_coordinates(parent)
        return child_coordinates[0][0] > parent_coordinates[0][0] \
               and child_coordinates[1][0] < parent_coordinates[1][0] \
               and child_coordinates[2][1] > parent_coordinates[2][1] \
               and child_coordinates[3][1] < parent_coordinates[3][1]

    @staticmethod
    def find_contour_coordinates(contour):
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        return leftmost, rightmost, topmost, bottommost

    @staticmethod
    def drawContours(contours, frame, color=(0, 0, 255)):
        for contour in contours:
            leftmost, rightmost, topmost, bottommost = Contours.find_contour_coordinates(contour)
            area = cv2.contourArea(contour)
            if area < 10000:  # Set a lower bound on the elipse area.
                continue

            cv2.drawContours(frame, contour, -1, color, 1)
            cv2.rectangle(frame, (leftmost[0], topmost[1]), (rightmost[0], bottommost[1]), (0, 255, 0), 2)
import cv2
def binarizeFrame(f):
    # binarizing image
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (19, 19), 0)  # Gaussian blur to reduce noise in the image.
    # Use adaptive thresholding to "binarize" the image.
    binarized = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    return binarized
import cv2
import numpy as np
import matplotlib.pyplot as plt


def counting(image, type):
    image = np.uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if type == 1:
        img_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 21)
        kernel = np.ones((13, 13), np.uint8)
        img = cv2.dilate(img_thresh, kernel)
        img = cv2.erode(img, kernel)
        img = cv2.medianBlur(img, 5)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_of_objects = len(contours)
        result = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)

    elif type == 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        # gray = cv2.equalizeHist(gray)

        gray = np.power(gray, 0.1)
        max_val = np.max(gray.ravel())
        gray = gray / max_val * 255
        gray = gray.astype(np.uint8)
        # Xử lý gợn sóng
        # fftGray = ft.fftshift(ft.fft2(gray))
        # magnitude_spectrum = 15 * np.log(np.abs(fftGray))

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -20)

        kernel = np.ones((3, 3), dtype=np.uint8)
        erode = cv2.erode(thresh, kernel)

        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        num_of_objects = len(contours)
        result = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)

    return {"outputImg": result, "count": num_of_objects}

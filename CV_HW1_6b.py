import numpy as np
import cv2


img = cv2.imread('chars1.png')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_image = cv2.threshold(gray_image, 117, 255, cv2.THRESH_BINARY_INV)[1]
found_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

list_of_widths, list_of_heights = zip(*[(cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[3])
                                        for contour in found_contours])

print(f"Количество символов: {len(found_contours)}")
print(f"Медианная ширина: {np.median(list_of_widths)}")
print(f"Медианная высота: {np.median(list_of_heights)}")


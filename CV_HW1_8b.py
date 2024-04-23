import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    return binary_image


def detect_lines(binary_image):
    height = binary_image.shape[1]
    masked_image = np.zeros_like(binary_image)
    masked_image[round(height * 0.2):, :] = binary_image[round(height * 0.2):, :]

    return cv2.HoughLines(masked_image, 10, np.pi / 165, 3700)


def draw_lines(image, lines):
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        point1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        point2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        cv2.line(image, point1, point2, (0, 255, 0), 3)


def find_intersection(lines):
    rho1, theta1 = lines[0][0]
    rho2, theta2 = lines[1][0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return int(np.round(x0)[0]), int(np.round(y0)[0])


def main(image_path):
    binary_image = load_and_preprocess_image(image_path)
    lines = detect_lines(binary_image)
    image = cv2.imread(image_path)
    draw_lines(image, lines)

    if len(lines) > 1:
        x0, y0 = find_intersection(lines)
        cv2.circle(image, (x0, y0), 5, (0, 0, 255), -1)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


main('road1.png')

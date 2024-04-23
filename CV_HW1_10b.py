import cv2
import numpy as np
import matplotlib.pyplot as plt


# Функция для нахождения углов шахматной доски на изображении
def find_chessboard_corners(image, pattern_size=(7, 5)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if found:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return found, corners


# Загрузим исходное изображение
src_image_path = 'src_balls.jpg'
src_image = cv2.imread(src_image_path)

# Ищем углы шахматной доски
found, corners = find_chessboard_corners(src_image)

# Если углы найдены, продолжаем с преобразованием перспективы
if found:
    # Сортируем углы шахматной доски (top-left, top-right, bottom-right, bottom-left)
    corners = corners.reshape(-1, 2)
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[6], corners[-1], corners[-7]

    # Определим целевые точки для перспективного преобразования
    # Мы знаем размеры шахматной доски, поэтому можем вычислить размер каждой клетки
    cell_size = 100  # размер каждой клетки на итоговом изображении
    width, height = 7 * cell_size, 5 * cell_size

    # Определим целевые точки для преобразованного изображения
    new_image_width, new_image_height = int(src_image.shape[0] * 1.5), int(src_image.shape[1] * 1.5)
    destination_points = np.array([
        [new_image_height - height, new_image_width - width], [new_image_height - height, new_image_width],
        [new_image_height, new_image_width], [new_image_height, new_image_width - width],
    ], dtype='float32')

    # Исходные точки на изображении
    source_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Получим матрицу преобразования перспективы
    M = cv2.getPerspectiveTransform(source_points, destination_points)

    # Преобразуем изображение для получения вида сверху
    top_down_view = cv2.warpPerspective(src_image, M, (new_image_height, new_image_width))
else:
    top_down_view = None
gray = cv2.cvtColor(top_down_view[:, :-1000], cv2.COLOR_BGR2GRAY)

# Получим бинарное изображение и на нём найдем и отрисуем контуры шаров и кия
retval, binary_image = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(binary_image, contours, -1, (0, 0, 255), 25)

# Найдем шары на изображении, чтобы взять их центры
circles = cv2.HoughCircles(
    binary_image,
    cv2.HOUGH_GRADIENT,
    dp=1,  # Отношение разрешения
    minDist=50,  # Минимальное расстояние между центрами обнаруженных кругов
    param1=610,  # Верхний порог для внутреннего процесса Canny edge detector
    param2=12,  # Пороговое значение для центра обнаружения этапа Хафа
    minRadius=130,  # Минимальный радиус круга, который будет обнаружен
    maxRadius=180  # Максимальный радиус
)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Рисование внешнего круга
        cv2.circle(top_down_view, (i[0], i[1]), i[2], (0, 0, 255), 15)
        # Рисование центра круга
        cv2.circle(top_down_view, (i[0], i[1]), 2, (0, 0, 255), 6)

cv2.drawContours(top_down_view, contours, -1, (0, 0, 255), 10)

target_ball_center, target_ball_radius = np.array([circles[0, 0][0], circles[0, 0][1]]), circles[0, 0][2]  # Центр прицельного шара
cue_ball_center, cue_ball_radius = np.array([circles[0, 1][0], circles[0, 1][1]]), circles[0, 1][2]  # Центр битого шара

# Находим контур с максимальной длиной, предполагая, что это контур кия
cue_contour = max(contours, key=cv2.contourArea)

# Применяем fitLine для получения параметров вектора линии
[vx, vy, x, y] = cv2.fitLine(cue_contour, cv2.DIST_L2, 0, 0.01, 0.01)

# Вычисляем две точки для отрисовки линии на исходном изображении
# Линия будет проходить через точку (x, y) и направлена вдоль вектора (vx, vy)
# Мы выбираем t так, чтобы линия хорошо вписывалась в размеры изображения
t = 1000  # Длина линии
pt1 = (int(x - vx * t), int(y - vy * t))  # Начало линии
pt2 = (int(x + vx * t), int(y + vy * t))  # Конец линии

x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]

# Координаты центра шара (cx, cy) и его радиус (r)
cx, cy, r = circles[0, 1]

# Переведем точки в numpy массивы
P1 = np.array([x1, y1])
P2 = np.array([x2, y2])
center = np.array([cx, cy])

# Нормализуем направляющий вектор линии
direction = P2 - P1
cue_direction = direction / np.linalg.norm(direction)

#ПОЧЕСТНОМУ БУДЕМ ВЫЧИСЛЯТЬ КУДА ПОЛЕТИТ ШАР (СПОЙЛЕР: он полетит в сторону)
# t = ((cx - x1) * cue_direction[0] + (cy - y1) * cue_direction[1])
# nearest_x = x1 + t * cue_direction[0]
# nearest_y = y1 + t * cue_direction[1]
# dist_to_line = np.sqrt((cx - nearest_x)**2 + (cy - nearest_y)**2)
# if dist_to_line < r:
#     # Находим длину отрезка от центра шара до точки касания
#     d = np.sqrt(r**2 - dist_to_line**2)
#     touch_x = nearest_x - d * cue_direction[0]
#     touch_y = nearest_y - d * cue_direction[1]
#
#     cv2.circle(top_down_view, (int(touch_x), int(touch_y)), 1, (0, 255, 0), 20)
#
#     # Рассчитываем направление движения шара
#     normal = np.array([touch_x - cx, touch_y - cy])
#     normal = normal / np.linalg.norm(normal)
#     ball_direction = 2 * (np.dot(cue_direction, normal)) * normal - cue_direction
#
#     # Отрисовываем новую траекторию
#     final_x = int(cx + ball_direction[0] * 10000)
#     final_y = int(cy + ball_direction[1] * 10000)
#     cv2.line(top_down_view, (cx, cy), (final_x, final_y), (0, 255, 0), 20)


#так как по честному не получилось сделаем по простому
# Найдем две точки на линии на расстоянии двух радиуса от центра шара
new_center = target_ball_center - cue_direction * 2*r
pt2 = target_ball_center + cue_direction * 10000


# Рисуем новый шар
cv2.circle(top_down_view, (int(new_center[0]), int(new_center[1])), r, (0, 255, 0), 20)
cv2.line(top_down_view, target_ball_center, (int(pt2[0]), int(pt2[1])), (0, 255, 0), 10)


# Визуализируем результат
if top_down_view is not None:
    plt.figure(figsize=(10, 10))
    plt.imshow(top_down_view)
    plt.title('Top-Down View')
    plt.show()
else:
    print("Не удалось найти углы шахматной доски.")

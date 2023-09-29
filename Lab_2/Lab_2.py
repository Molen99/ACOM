import cv2
import numpy as np

# Задание 1: Чтение изображения с камеры
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)  # 0 - индекс камеры (обычно это встроенная камера)

while True:
    ret, frame = cap.read()  # Чтение кадра с камеры
    if not ret:
        break

# Задание 2: Фильтрация и вывод красной части изображения
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    red_mask = cv2.inRange(frame, lower_red, upper_red)
    red_filtered = cv2.bitwise_and(frame, frame, mask=red_mask)

# Задание 3: Морфологические преобразования
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(red_filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Задание 4: Преобразование в черно-белое изображение и поиск моментов
    gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Площадь объекта:", area)

# Задание 5: Найти центр и построить прямоугольник
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.rectangle(frame, (cx - 50, cy - 50), (cx + 50, cy + 50), (0, 0, 0), 2)

    cv2.imshow("Original", frame)
    cv2.imshow("Filtered", closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
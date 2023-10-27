import cv2
import numpy as np
import math

file_name = 'img.png'


def task2():
    image1 = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('Window 1', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Window 1', image1)
    cv2.waitKey(0)

    image2 = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cv2.namedWindow('Window 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Window 2', image2)
    cv2.waitKey(0)

    image3 = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    cv2.namedWindow('Window 3', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Window 3', image3)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def task3():
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task4():

    input_video_path = 'video.mp4'
    cap = cv2.VideoCapture(input_video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (w, h))

    while True:
        ret, frame = cap.read()
        out.write(frame)

        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def task5():
    image = cv2.imread(file_name)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imshow('Original Image', image)
    cv2.imshow('HSV Image', hsv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_pentagram_with_circle(image, center, size, color, thickness):
    points = []
    for i in range(5):
        x = int(center[0] + size * math.cos(2 * math.pi * i / 5))
        y = int(center[1] + size * math.sin(2 * math.pi * i / 5))
        points.append((x, y))
    for i in range(5):
        cv2.line(image, points[i], points[(i + 2) % 5], color, thickness)

    # Рисование круга в центре пентаграммы
    cv2.circle(image, center, size // 1, color, thickness)


def task6():
    input_video_path = 'video.mp4'
    cap = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape

        pentagram_image = np.copy(frame)
        center = (width // 2, height // 2)
        size = 100  # Размер пентаграммы
        color = (0, 0, 255)  # Красный цвет
        thickness = 15  # Толщина линий

        draw_pentagram_with_circle(pentagram_image, center, size, color, thickness)

        cv2.imshow('Pentagram with Circle', pentagram_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task8():
    input_video_path = 'video.mp4'
    cap = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        # Преобразование цвета центрального пикселя в HSV
        hsv_color = cv2.cvtColor(np.uint8([[frame[center_y, center_x]]]), cv2.COLOR_BGR2HSV)[0][0]

        if hsv_color[0] >= 0 and hsv_color[0] < 20:
            # Ближе к красному (по HSV)
            fill_color = (0, 0, 255)  # Красный
        elif hsv_color[0] >= 40 and hsv_color[0] < 80:
            # Ближе к зеленому (по HSV)
            fill_color = (0, 255, 0)  # Зеленый
        else:
            # Ближе к синему (по HSV)
            fill_color = (255, 0, 0)  # Синий

        cv2.rectangle(frame, (width // 2 - 100, height // 2 - 20), (width // 2 + 100, height // 2 + 20),
                      fill_color, 15)
        cv2.rectangle(frame, (width // 2 - 20, height // 2 - 100), (width // 2 + 20, height // 2 + 100),
                      fill_color, 15)

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task9():
    video = cv2.VideoCapture("http://192.168.1.42:8080/video")
    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("outputcamtel.mov", fourcc, 25, (w, h))
    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # task2()
    # task3()
    # task4()
    # task5()
     task6()
    # task8()
    # task9()

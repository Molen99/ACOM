import cv2
import numpy as np


def haar_cascade(video):
    # Загрузка каскада Хаара для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Запуск видеопотока с камеры
    cap = cv2.VideoCapture(video)

    while True:
        # Захват кадра с камеры
        ret, frame = cap.read()

        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Рисование прямоугольников вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Отображение результата
        cv2.imshow('Face Detection', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


def yolo(video):
    # Загрузка конфигурации и весов модели YOLO
    net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

    # Загрузка классов
    classes = []
    with open("face.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    # Захват видеопотока с камеры (0 - камера по умолчанию)
    cap = cv2.VideoCapture(video)

    while True:
        # Захват кадра
        _, frame = cap.read()

        # Получение высоты и ширины кадра
        height, width, _ = frame.shape

        # Преобразование кадра в blob, чтобы его можно было передать в YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Получение выходных слоев YOLO
        outs = net.forward(layer_names)

        # Парсинг результатов
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Проверка, что обнаружен объект - лицо
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Удаление лишних прямоугольников
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        scale_factor = 0.6

        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                x += 150
                y += 50
                w = int(w * scale_factor)
                h = int(h * scale_factor)
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Зеленый цвет
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # for i in range(len(boxes)):
        #     if i in indices:
        #         x, y, w, h = boxes[i]
        #         label = str(classes[class_ids[i]])
        #         confidence = confidences[i]
        #         color = (0, 255, 0)  # Зеленый цвет
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #         cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Отображение результата
        cv2.imshow("Face Detection", frame)

        # Прерывание при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


haar_cascade(0)
yolo(0)
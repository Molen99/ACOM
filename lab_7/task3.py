from difflib import SequenceMatcher

import pandas as pd
import cv2
from pytesseract import image_to_string
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

labels = ["croaked растаял", "величии лишил", "Вот он ваш знак", "гидов лавки", "дружбу cloudless", "застыло безопасно",
          "земляного родилась", "охотно нанял", 'серьги корону']
special_symbol = "".join(['-' for _ in range(100)])


def rotate_and_recognize(image_path):
    # Загружаем изображение
    img = cv2.imread(image_path)

    # Перебираем углы поворота от -20 до 20 с шагом 1
    all_results = []
    for angle in range(-20, 21):
        # Поворачиваем изображение
        rotated_img = rotate_image(img, angle)

        # Применяем Tesseract OCR
        result = image_to_string(rotated_img, lang='rus+eng')
        all_results.append(result)

    # Сформируем итоговый ответ на основе результатов для каждого угла поворота
    final_result = " ".join(all_results)

    return final_result


def rotate_image(image, angle):
    # Получаем размеры изображения
    height, width = image.shape[:2]

    # Вычисляем центр изображения
    center = (width // 2, height // 2)

    # Получаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Поворачиваем изображение
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def calculate_accuracy_percentage(generated_text, expected_text):
    # Используем SequenceMatcher для сравнения строк посимвольно
    matcher = SequenceMatcher(None, generated_text, expected_text)
    match_percentage = matcher.ratio() * 100
    return match_percentage


def test_rotation_and_recognition(dataset_folder):
    output_file = "result_rotation_and_recognition.txt"
    is_correct_list = []

    with open(output_file, 'w', encoding="utf-8") as result_file:
        for image_filename in os.listdir(dataset_folder):
            index = int(str(image_filename)[0]) - 1
            print(index)

            image_path = os.path.join(dataset_folder, image_filename)

            # Получаем итоговый ответ с использованием Tesseract OCR
            recognized_text = rotate_and_recognize(image_path)

            expected_text = labels[index]

            # Сравниваем результат с ожидаемым текстом
            is_correct = int(calculate_accuracy_percentage(recognized_text, expected_text))

            # Записываем результат в файл
            result_file.write(
                f"Исходник: {expected_text} || Распознано: {recognized_text} || Корректность: {is_correct}\n{special_symbol}\n")

            is_correct_list.append(is_correct)

    return is_correct_list


if __name__ == "__main__":
    original_dataset_folder = "cap"

    # Тестирование нового метода на исходном датасете
    rotation_and_recognition_results = test_rotation_and_recognition(original_dataset_folder)

    # Запись результатов тестирования в сводную таблицу
    results_df = pd.DataFrame({"Rotation and Recognition": rotation_and_recognition_results})
    results_df.to_csv("rotation_and_recognition_results.csv", index=False)

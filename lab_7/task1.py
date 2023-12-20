from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from pytesseract import image_to_string
import pytesseract
import easyocr
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

labels = ["croaked растаял", "величии лишил", "Вот он ваш знак", "гидов лавки", "дружбу cloudless", "застыло безопасно",
          "земляного родилась", "охотно нанял", 'серьги корону']
special_symbol = "".join(['-' for _ in range(100)])


def straight_recognition(image_path, rec_type):
    if rec_type == "tesseract":
        return image_to_string(image_path, lang='rus+eng')
    elif rec_type == "easyOCR":
        try:
            reader = easyocr.Reader(['ru'])

            # Преобразование пути к файлу в объект Path
            image_path = Path(image_path)

            # Чтение изображения с использованием cv2
            img = cv2.imread(str(image_path))

            if img is None:
                raise Exception(f"Unable to read the image: {image_path}")

            result = reader.readtext(img)
            recognized_text = ''
            result.sort(key=lambda x: x[0][0])  # Сортируем по координатам X
            recognized_text = ' '.join([i[1] for i in result])

            return recognized_text
        except Exception as e:
            print(f"Error during easyOCR recognition: {e}")
            return ""


def exact_match(generated_text, expected_text):
    return generated_text == expected_text


def calculate_accuracy_percentage(generated_text, expected_text):
    # Используем SequenceMatcher для сравнения строк посимвольно
    matcher = SequenceMatcher(None, generated_text, expected_text)
    match_percentage = matcher.ratio() * 100
    return match_percentage


def test_recognition(rec_type, val_type):
    input_folder = "cap"
    output_file = f"result_{rec_type}_{val_type}.txt"

    # Открываем файл для записи результатов
    with open(output_file, 'w') as result_file:
        for image_filename in os.listdir(input_folder):
            index = int(str(image_filename)[0]) - 1
            print(index)

            image_path = os.path.join(input_folder, image_filename)

            # Получаем распознанный текст в зависимости от rec_type
            recognized_text = straight_recognition(image_path, rec_type)

            expected_text = labels[index]

            if val_type == "exact_match":
                correct = exact_match(recognized_text, expected_text)

            elif val_type == "character_accuracy":
                accuracy_percentage = int(calculate_accuracy_percentage(recognized_text, expected_text))
                # is_correct = accuracy_percentage > 90
                is_correct = str(accuracy_percentage) + "%"

            # Записываем результат в файл
            result_file.write(
                f"Исходник: {expected_text} || Распознано: {recognized_text} || Корректность: {is_correct}\n{special_symbol}\n")

    return is_correct


if __name__ == "__main__":
    recognition_type = "easyOCR"
    validation_type = "character_accuracy"
    test_recognition(recognition_type, validation_type)

import re
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


def calculate_accuracy_percentage(generated_text, expected_text):
    # Используем SequenceMatcher для сравнения строк посимвольно
    matcher = SequenceMatcher(None, generated_text, expected_text)
    match_percentage = matcher.ratio() * 100
    return match_percentage


def post_process_text(text):
    # Удаляем спецсимволы и приводим к одному регистру
    cleaned_text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text).lower()
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text


def recognize_and_post_process(image_path):
    # Загружаем изображение
    img = cv2.imread(image_path)

    # Применяем Tesseract OCR
    recognized_text = image_to_string(img, lang='rus+eng')

    # Постобработка текста
    processed_text = post_process_text(recognized_text)

    return processed_text


def test_recognition_and_post_process(dataset_folder):
    output_file = f"result_recognition_and_post_process_{dataset_folder}.txt"
    is_correct_list = []

    with open(output_file, 'w', encoding="utf-8") as result_file:
        for image_filename in os.listdir(dataset_folder):
            index = int(str(image_filename)[0]) - 1
            print(index)

            image_path = os.path.join(dataset_folder, image_filename)

            # Получаем итоговый ответ с использованием Tesseract OCR и постобработку
            recognized_text = recognize_and_post_process(image_path)

            expected_text = labels[index]

            # Сравниваем результат с ожидаемым текстом
            is_correct = int(calculate_accuracy_percentage(recognized_text, expected_text))

            #is_correct = str(is_correct) + "%"

            # Записываем результат в файл
            result_file.write(
                f"Исходник: {expected_text} || Распознано: {recognized_text} || Корректность: {is_correct}\n{special_symbol}\n")

            is_correct_list.append(is_correct)

    return is_correct_list


if __name__ == "__main__":
    original_dataset_folder = "cap"

    # Тестирование нового метода на исходном датасете
    recognition_and_post_process_results_original = test_recognition_and_post_process(original_dataset_folder)

    # Запись результатов тестирования в сводную таблицу
    results_original_df = pd.DataFrame(
        {"Recognition and Post Process (Original Dataset)": recognition_and_post_process_results_original})
    results_original_df.to_csv("recognition_and_post_process_results_original.csv", index=False)

    # Тестирование нового метода на датасете2
    dataset2_folder = "cap_aug"
    recognition_and_post_process_results_dataset2 = test_recognition_and_post_process(dataset2_folder)

    # Запись результатов тестирования на датасете2 в сводную таблицу
    results_dataset2_df = pd.DataFrame(
        {"Recognition and Post Process (Dataset2)": recognition_and_post_process_results_dataset2})
    results_dataset2_df.to_csv("recognition_and_post_process_results_dataset2.csv", index=False)

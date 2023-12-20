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


def boxes_recognition(dataset_folder):
    output_file = f"task6_{dataset_folder}.txt"
    is_correct_list = []

    with open(output_file, 'w', encoding="utf-8") as result_file:
        for image_filename in os.listdir(dataset_folder):
            index = int(str(image_filename)[0]) - 1
            print(index)

            image_path = os.path.join(dataset_folder, image_filename)
            img = cv2.imread(image_path, 0)

            h, w = img.shape
            boxes = pytesseract.image_to_boxes(img, lang="rus+eng")

            for box in boxes.splitlines():
                box_data = box.split(" ")
                cv2.rectangle(
                    img,
                    (int(box_data[1]), h - int(box_data[2])),
                    (int(box_data[3]), h - int(box_data[4])),
                    (0, 255, 0),
                    2,
                )
            expected_text = labels[index]
            is_correct = int(
                calculate_accuracy_percentage("".join([sym_data.split(" ")[0] for sym_data in boxes.split("\n")]),
                                              expected_text))

            result = "".join([sym_data.split(" ")[0] for sym_data in boxes.split("\n")])

            result_file.write(
                f"Исходник: {expected_text} || Распознано: {result} || Корректность: {is_correct}\n{special_symbol}\n")

            is_correct_list.append(is_correct)

    return is_correct_list


if __name__ == "__main__":
    original_dataset_folder = "cap"
    # Тестирование нового метода на исходном датасете
    recognition_and_post_process_results_original = boxes_recognition(original_dataset_folder)

    # Запись результатов тестирования в сводную таблицу
    results_original_df = pd.DataFrame(
        {"Recognition and Post Process (Original Dataset)": recognition_and_post_process_results_original})
    results_original_df.to_csv("1original.csv", index=False)

    # Тестирование нового метода на датасете2
    dataset2_folder = "cap_aug"
    recognition_and_post_process_results_dataset2 = boxes_recognition(dataset2_folder)

    # Запись результатов тестирования на датасете2 в сводную таблицу
    results_dataset2_df = pd.DataFrame(
        {"Recognition and Post Process (Dataset2)": recognition_and_post_process_results_dataset2})
    results_dataset2_df.to_csv("1dataset2.csv", index=False)

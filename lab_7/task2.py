from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
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


def test_recognition(rec_type, val_type, dataset_folder):
    output_file = f"result_{rec_type}_{val_type}_{dataset_folder}.txt"
    index = 0
    is_correct_list = []
    result_str = ''

    with open(output_file, 'w', encoding="utf-8") as result_file:
        for image_filename in os.listdir(dataset_folder):
            index = int(str(image_filename)[0]) - 1
            print(index)

            image_path = os.path.join(dataset_folder, image_filename)

            recognized_text = straight_recognition(image_path, rec_type)
            expected_text = labels[index]

            if val_type == "exact_match":
                is_correct = exact_match(recognized_text, expected_text)
            elif val_type == "character_accuracy":
                accuracy_percentage = int(calculate_accuracy_percentage(recognized_text, expected_text))
                is_correct = str(accuracy_percentage) + "%"

            is_correct_list.append(is_correct)
            result_str += f"Исходник: {expected_text} || Распознано: {recognized_text} || Корректность: {is_correct}\n{special_symbol}\n"

        result_file.write(result_str)
    return is_correct_list


def augment_dataset(input_folder, output_folder):
    # Создаем выходную папку, если её нет
    os.makedirs(output_folder, exist_ok=True)

    # Перебираем изображения в исходной папке
    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)

        # Загружаем изображение
        img = cv2.imread(image_path)

        # Перебираем углы поворота от -20 до 20 с шагом 1
        for angle in range(-20, 21):
            # Поворачиваем изображение
            rotated_img = rotate_image(img, angle)

            # Формируем новое имя файла, чтобы отразить поворот
            new_filename = f"{os.path.splitext(image_filename)[0]}_rotated_{angle}.png"
            new_image_path = os.path.join(output_folder, new_filename)

            # Сохраняем повернутое изображение
            cv2.imwrite(new_image_path, rotated_img)


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


if __name__ == "__main__":
    original_dataset_folder = "cap"
    augmented_dataset_folder = "cap_aug"
    # augment_dataset(original_dataset_folder, augmented_dataset_folder)

    recognition_type = "tesseract"
    validation_type = "character_accuracy"

    # original_results = test_recognition(recognition_type, validation_type, original_dataset_folder)

    # Запись результатов тестирования на исходном датасете в сводную таблицу
    # original_results_df = pd.DataFrame({"Original Dataset": original_results})
    # original_results_df.to_csv("original_results.csv", index=False)

    # Аугментация датасета и тестирование на измененном датасете
    augment_dataset(original_dataset_folder, augmented_dataset_folder)
    augmented_results = test_recognition(recognition_type, validation_type, augmented_dataset_folder)

    # Запись результатов тестирования на измененном датасете в сводную таблицу
    augmented_results_df = pd.DataFrame({"Augmented Dataset": augmented_results})
    augmented_results_df.to_csv("augmented_results.csv", index=False)

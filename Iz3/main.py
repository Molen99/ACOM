import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from scipy import ndimage


def get_model(width=128, height=128, depth=64):
    """Обновленная модель трехмерной сверточной нейронной сети."""
    inputs = keras.Input((width, height, depth, 1))

    # Слой свертки с увеличенным dilation_rate для увеличения "receptive field"
    x = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", dilation_rate=(3, 3, 3))(inputs)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    # Слой свертки с измененными параметрами
    x = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", dilation_rate=(2, 2, 2))(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    # Слой свертки с измененными параметрами
    x = keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu", dilation_rate=(2, 2, 2))(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    # Слой свертки без изменений
    x = keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    # Глобальное усреднение по пространственным измерениям
    x = keras.layers.GlobalAveragePooling3D()(x)

    # Выпрямление для подачи в полносвязные слои
    x = keras.layers.Flatten()(x)

    # Полносвязные слои с измененными параметрами
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)

    # Слой Dropout для регуляризации
    x = keras.layers.Dropout(0.2)(x)

    # Выходной слой
    outputs = keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Определение модели.
    model = keras.Model(inputs, outputs, name="3dcnn_updated")
    return model


@tf.function
def rotate(volume):
    """Поворот объема на несколько градусов"""

    def scipy_rotate(volume):
        # Определение углов поворота
        angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        # Выбор случайного угла
        angle = random.choice(angles)
        # Поворот объема
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    # Поворот объема
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def main():
    print("Количество доступных GPU: ", len(tf.config.list_physical_devices('GPU')))

    with open('Vul.pickle', 'rb') as file:
        loaded_state = pickle.load(file)

    abnormal_scans = loaded_state['abnormal_scans']
    normal_scans = loaded_state['normal_scans']

    abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
    normal_labels = np.array([0 for _ in range(len(normal_scans))])

    print(f'Длина аномальных сканов: {len(abnormal_scans)}')
    print(f'Длина нормальных сканов: {len(normal_scans)}')

    # Построение модели.
    model = get_model(width=128, height=128, depth=64)

    # Разделение данных в соотношении 70-30 для обучения и валидации.
    x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
    y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
    x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
    y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
    print(
        "Количество примеров в обучающем и валидационном наборах данных %d и %d."
        % (x_train.shape[0], x_val.shape[0])
    )

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    print('Начало аугментации')
    batch_size = 2
    # Аугментация данных в процессе обучения.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    print('Предобработка валидации')
    # Только масштабирование.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    # Обучение -------------------------------------------------------------------------------------------------
    print('Начало обучения')
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Определение коллбэков.
    # Лучшая модель будет сохранена в этот файл
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "Luxs.h5", save_best_only=True, monitor="val_acc"
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc",
                                                      patience=40)

    # Обучение модели с валидацией в конце каждой эпохи
    epochs = 40
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    model.save('base_model', save_format='tf')


if __name__ == '__main__':
    main()

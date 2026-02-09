import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from database_manager import setup_database, clear_student_data, add_new_student, load_known_faces



IMAGE_SIZE = (96, 96)
MODEL_NAME = 'class_face_model.h5'
BATCH_SIZE = 32
EPOCHS = 45
TRAIN_DATA_DIR = '../class_photos_to_register'
DB_FILE_NAME = '../new_class_data.db'


#архітектура мережі
def build_custom_cnn(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#основна ф-ція навчання
def train_new_class_model():
    # 1. Створення/очищення бази даних
    setup_database(DB_FILE_NAME)

    clear_student_data(DB_FILE_NAME)

    # Підготовка генераторів даних
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    #  Завантаження даних із папок

    train_generator = datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    #  Перевірка кількості класів та фотографій
    num_classes = train_generator.num_classes
    if num_classes == 0 or train_generator.samples < num_classes * 5:
        print(
            "📢 Помилка: Знайдено замало класів або фотографій. Перевірте, чи правильно заповнена папка 'class_pohoto_to_register'!")
        print(f"Знайдено класів: {num_classes}, Знайдено фото: {train_generator.samples}")
        return  # Вихід, якщо немає даних

    print(f"✅ Починаємо навчання НМ. Класів: {num_classes}, Зразків: {train_generator.samples}")

    # Створення та навчання моделі
    model = build_custom_cnn(num_classes)
    model.summary()

    # Збереження імен та ID у базу даних
    for full_name_with_index, index in train_generator.class_indices.items():

        clean_full_name = re.sub(r'\s\d+$', '', full_name_with_index)


        parts = clean_full_name.split('_')
        first_name = parts[0].strip()
        last_name = parts[1].strip() if len(parts) > 1 else ""


        class_index_encoding = np.array([index], dtype=np.float64).tobytes()

        # Виклик функції збереження
        add_new_student(first_name, last_name, index, DB_FILE_NAME)

    #  Навчання
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        epochs=EPOCHS
    )

    # ГРАФІК
    plt.figure(figsize=(10, 6))

    # Малюємо Accuracy
    plt.plot(history.history['accuracy'], color='tab:blue', linewidth=2, label='Accuracy (Точність)')

    # Малюємо Loss
    plt.plot(history.history['loss'], color='tab:red', linewidth=2, label='Loss (Помилка)')

    # Налаштування осей та сітки
    plt.title('Метрики навчання моделі MyClassAI')
    plt.xlabel('Епоха')
    plt.ylabel('Значення показників')  # Загальний підпис для обох
    plt.grid(True, linestyle='--', alpha=0.6)


    plt.legend(loc='upper left')


    plt.ylim(bottom=0)

    # Збереження
    plot_filename = 'training_report.png'
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"📊 Об'єднаний графік (одна шкала) збережено як '{plot_filename}'")

    # 8. Збереження моделі
    model.save(MODEL_NAME)
    print(f" Навчання завершено! Модель збережено як '{MODEL_NAME}'.")

    # 9. Фінальна перевірка
    known_face_ids, known_face_names = load_known_faces(DB_FILE_NAME)
    print(" База даних учнів оновлена:")
    print(known_face_names)


if __name__ == '__main__':
    train_new_class_model()
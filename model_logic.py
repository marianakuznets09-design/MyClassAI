import random
import sqlite3
from database_manager import update_chosen_count
import numpy as np



DATABASE_NAME = 'new_class_data.db'


def get_weighted_choice(present_class_indices):
    """
    Обирає учня на основі зваженої вірогідності. Хто рідше виходив, має більший шанс.
    Вибір відбувається ЛИШЕ серед тих учнів, чиї індекси є у present_class_indices.

    :param present_class_indices: Список class_index (цілі числа), розпізнаних у поточному кадрі.
    :return: (id, full_name, class_index) або (None, повідомлення, None)
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute('SELECT id, first_name, last_name, times_chosen, is_immune, face_encoding FROM Students')
    students_data = cursor.fetchall()
    conn.close()

    if not students_data:
        return None, "Немає учнів у базі даних.", None



    available_students = []

    for row in students_data:
        # id=0, first_name=1, last_name=2, times_chosen=3, is_immune=4, face_encoding=5
        db_id, first_name, last_name, times_chosen, is_immune, face_encoding_bytes = row


        try:
            class_index = int(np.frombuffer(face_encoding_bytes, dtype=np.float64)[0])
        except Exception:

            continue

            # 1. Перевірка на присутність у кадрі
        if class_index not in present_class_indices:
            continue  # Пропустити, якщо не розпізнаний

        # 2. Перевірка на імунітет
        if is_immune:
            continue

        # Якщо присутній та не має імунітету, додаємо його до списку доступних
        available_students.append({
            'id': db_id,
            'name': f"{first_name} {last_name}".strip(),
            'times': times_chosen,
            'index': class_index
        })

    if not available_students:
        return None, "Немає доступних учнів (нікого не видно або всі імунні).", None

    #2

    # Знаходимо мінімальну кількість виходів
    min_times = min(s['times'] for s in available_students)

    weights = []
    # Параметр випадковості: 0.15 означає
    randomness_factor = 0.15

    for student in available_students:
        relative_times = student['times'] - min_times

        # алгоритмічна частина
        base_weight = 100.0 / ((relative_times + 1) ** 2)


        final_weight = (base_weight * (1 - randomness_factor)) + (random.random() * randomness_factor * 100)

        weights.append(final_weight)

    # 3. Вибір респондента
    total_weight = sum(weights)
    if total_weight == 0:
        chosen_index_idx = random.randrange(len(available_students))
    else:
        # математика + рандом
        chosen_index_idx = random.choices(range(len(available_students)), weights=weights, k=1)[0]

    #3

    chosen_student = available_students[chosen_index_idx]
    chosen_id = chosen_student['id']
    chosen_name = chosen_student['name']
    chosen_index = chosen_student['index']


    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE Students SET times_chosen = times_chosen + 1 WHERE id = ?', (chosen_id,))
    conn.commit()
    conn.close()

    return chosen_id, chosen_name, chosen_index

# DATABASE_NAME = 'new_class_data.db'
#
#
# def get_weighted_choice():
#     """Обирає учня на основі зваженої вірогідності. Хто рідше виходив, має більший шанс."""
#     conn = sqlite3.connect(DATABASE_NAME)
#     cursor = conn.cursor()
#     cursor.execute('SELECT id, first_name, last_name, times_chosen, is_immune, face_encoding FROM Students')
#     students_data = cursor.fetchall()
#     conn.close()
#
#     if not students_data:
#         return None, "Немає учнів у базі даних.", None
#
#     active_students_weights = []
#
#     for id, first_name, last_name, times_chosen, is_immune, encoding_blob in students_data:
#
#         if is_immune == 1:
#             weight = 0
#         else:
#             weight = 1.0 / (times_chosen + 1)
#
#         # Отримуємо class_index з БД
#         class_index = int(np.frombuffer(encoding_blob, dtype=np.float64)[0])
#
#         active_students_weights.append({
#             'id': id,
#             'name': first_name,
#             'weight': weight,
#             'class_index': class_index  # Повертаємо індекс НМ
#         })
#
#     active_population = [s for s in active_students_weights if s['weight'] > 0]
#     weights = [s['weight'] for s in active_students_weights if s['weight'] > 0]
#
#     if sum(weights) == 0:
#         return None, "Всі активні учні мають нульову вагу або всі учні імунні.", None
#
#     # Зважений вибір об'єкта
#     chosen_student = random.choices(active_population, weights=weights, k=1)[0]
#
#     # Оновлення статистики
#     update_chosen_count(chosen_student['id'])
#
#     # Повертаємо ID та ім'я, а також class_index для розпізнавання
#     return chosen_student['id'], chosen_student['name'], chosen_student['class_index']
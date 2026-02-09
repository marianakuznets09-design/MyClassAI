from flask import Flask, render_template, request, redirect, url_for, Response  # Додано Response
from database import setup_db, add_class_to_db, get_all_classes, delete_class_from_db
from database import add_student_to_db, get_students_by_class
import cv2
import numpy as np
import random
import shutil
from flask import request, redirect, url_for
import os
import time
import subprocess
import math
import pandas as pd
from flask import send_file
from tensorflow.keras.models import load_model # Це для завантаження ШІ-моделі
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from PIL import ImageFont, ImageDraw, Image
camera_active = False
final_frame = None
detected_today = [] # Список для збереження тих, кого побачив ШІ
selected_student_name = ""
confidence = 0.0
is_immunity_searching = False
immunity_search_start = 0
immunity_winner = None
app = Flask(__name__)

last_recognized_student = None
last_recognition_time = 0
current_present_students = []
student_counts = {}
immune_students = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Завантажуємо нейромережу

try:
    model = load_model('class_face_model.h5')
except:
    print("Помилка: Файл моделі class_face_model.h5 не знайдено!")

#  Список імен
class_names = ['Учень 1', 'Учень 2', 'Учень 3']
student_counts = {name: 0 for name in class_names}
#  Глобальні змінні для логіки вибору
is_searching = False
selected_student_name = ""
search_counter = 0

is_searching = False
search_start_time = 0
selected_face_index = -1
detected_faces_coords = []
winner_name = ""

# Ініціалізація бази при запуску
setup_db()


def get_current_students():

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_photos_to_register")

    if os.path.exists(base_path):

        return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return []
# ЛОГІКА ДЛЯ КАМЕРИ
@app.route('/toggle_camera_status/<status>')
def toggle_camera_status(status):
    global camera_active
    camera_active = (status == 'true')
    return "OK"


camera = cv2.VideoCapture(0)

# роздільну здатність для стабільності
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)






face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def gen_frames():
    global is_searching, search_start_time, selected_face_index, winner_name
    global last_recognized_student, last_recognition_time, student_counts



    while True:
        success, frame = camera.read()
        if not success:
            continue


        # if is_immunity_searching:
        #     if time.time() - immunity_search_start < 3:  # Анімація 3 сек
        #         if len(faces) > 0:
        #             # Обираємо випадкове обличчя для "стрибка" кружечка
        #             f = faces[random.randint(0, len(faces) - 1)]
        #             (x, y, w, h) = f
        #             # Малюємо ніжний рожевий кружечок над головою
        #             cv2.circle(frame, (x + w // 2, y - 40), 15, (203, 192, 255), -1)
        #     else:
        #         is_immunity_searching = False

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        detected_faces_coords = faces  # Запам'ятовуємо координати всіх облич

        current_time = time.time()



        for (x, y, w, h) in faces:
            detected_name = "Unknown"
            try:
                #  Підготовка зображення
                face_img = frame[y:y + h, x:x + w]

                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img_resized = cv2.resize(face_img_rgb, (96, 96))

                # Нормалізація та зміна розмірності для моделі
                face_img_final = face_img_resized.astype("float32") / 255.0
                face_img_final = np.expand_dims(face_img_final, axis=0)

                # 2. Прогноз моделі
                prediction = model.predict(face_img_final, verbose=0)
                i = np.argmax(prediction[0])
                conf = prediction[0][i]

                #  Визначаємо ім'я
                if conf > 0.8:

                    students_list = get_current_students()
                    if i < len(students_list):
                        detected_name = students_list[i]
                        if detected_name not in current_present_students:
                            current_present_students.append(detected_name)
                            print(f"Присутній: {detected_name}")
                    else:
                        detected_name = "Unknown"



                # Малюємо рамку та ім'я для перевірки
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(frame, f"{detected_name} {int(conf * 100)}%", (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except Exception as e:
                print(f"Помилка в циклі розпізнавання: {e}")
                continue

            # if detected_name not in current_present_students:
            #     current_present_students.append(detected_name)
            #     print(f"Додано до присутніх: {detected_name}")
            if detected_name != "Unknown":
                current_time = time.time()
                # Захист від подвійного спрацювання (пауза 10 секунд між виходами одного учня)
                if detected_name != last_recognized_student or (current_time - last_recognition_time > 10):

                    # Перевірка на імунітет: якщо є імунітет, вихід НЕ зараховується
                    if detected_name not in immune_students:
                        if detected_name in student_counts:
                            student_counts[detected_name] += 1
                        else:
                            student_counts[detected_name] = 1

                        last_recognized_student = detected_name
                        last_recognition_time = current_time
                        print(f"Вихід зараховано: {detected_name}. Всього: {student_counts[detected_name]}")
            # Логіка анімації вибору (ваш попередній код)
        if is_searching:
            if time.time() - search_start_time < 3:
                if len(faces) > 0:
                    idx = random.randint(0, len(faces) - 1)
                    (x, y, w, h) = faces[idx]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
            else:
                is_searching = False
                if len(faces) > 0:
                    selected_face_index = random.randint(0, len(faces) - 1)
                    winner_name = "Обраний учень"

            # Малювання фінального результату
        if not is_searching and selected_face_index != -1 and len(faces) > selected_face_index:
            (xf, yf, wf, hf) = faces[selected_face_index]
            cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (0, 255, 0), 6)

            # --- ПРАВИЛЬНИЙ ВИВІД УКРАЇНСЬКОГО ТЕКСТУ ---
            # 1. Конвертуємо кадр OpenCV (BGR) у формат Pillow (RGB)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 2. Вказуємо шлях до шрифту (Windows стандарт) та розмір
            # Якщо у вас MacOS/Linux, шлях буде іншим (наприклад, /usr/share/fonts/...)
            font = ImageFont.truetype("arial.ttf", 32)

            # 3. Малюємо текст
            draw.text((xf, yf - 40), winner_name, font=font, fill=(0, 255, 0))

            # 4. Повертаємо назад у формат OpenCV
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------

@app.route('/')
def index():
    classes = get_all_classes()
    return render_template('index.html', classes=classes)


@app.route('/add_class', methods=['POST'])
def add_class():
    name = request.form.get('class_name')
    if name:
        add_class_to_db(name)
    return redirect(url_for('index'))


@app.route('/delete_class/<name>')
def delete_class(name):
    delete_class_from_db(name)
    return redirect(url_for('index'))


@app.route('/class/<class_name>')
def class_details(class_name):
    students_list = get_current_students()
    # Додайте student_counts=student_counts у return
    return render_template('class_details.html',
                           class_name=class_name,
                           students=students_list,
                           student_counts=student_counts)


def get_base_path():
    # Отримуємо папку, де лежить сам файл app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Додаємо назву папки з фото.
    # Якщо папка лежить прямо в проекті, залишаємо так.
    return os.path.join(base_dir, "class_photos_to_register")


@app.route('/add_student/<class_name>', methods=['POST'])
def add_student(class_name):
    student_name = request.form.get('student_name')
    if student_name:
        path = os.path.join(get_base_path(), student_name)
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Успішно створено папку для: {student_name}")
            else:
                print(f"Папка для {student_name} вже існує")
        except Exception as e:
            print(f"Помилка при додаванні: {e}")

    return redirect(url_for('edit_class', class_name=class_name))


@app.route('/delete_student/<class_name>/<student_name>', methods=['POST'])
def delete_student(class_name, student_name):
    path = os.path.join(get_base_path(), student_name)

    print(f"DEBUG: Спроба видалити -> {path}")

    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Успішно видалено папку: {student_name}")
        else:
            print(f"Помилка: Папка не знайдена за шляхом {path}")
    except Exception as e:
        print(f"Критична помилка видалення: {e}")

    return redirect(url_for('edit_class', class_name=class_name))

@app.route('/start_selection')
def start_selection():
    global is_searching, selected_student_name, search_counter
    is_searching = True
    search_counter = 40  # Скільки кадрів буде "стрибати" квадратик (приблизно 2-3 секунди)
    selected_student_name = ""
    return {"status": "started"}

@app.route('/edit_class/<class_name>')
def edit_class(class_name):
    # Використовуємо ту саму функцію
    students_list = get_current_students()
    return render_template('edit_class.html', class_name=class_name, students=students_list)


@app.route('/upload_photo/<class_name>/<student_name>', methods=['POST'])
def upload_photo(class_name, student_name):
    if 'photo' not in request.files:
        print("Файли не вибрано")
        return redirect(request.url)

    # Отрим. список усіх файлів
    files = request.files.getlist('photo')

    base_path = get_base_path()
    student_dir = os.path.join(base_path, student_name)

    if not os.path.exists(student_dir):
        os.makedirs(student_dir)

    saved_count = 0
    for file in files:
        if file and file.filename != '':

            filename = secure_filename(file.filename)

            unique_name = f"{int(time.time())}_{filename}"

            file.save(os.path.join(student_dir, unique_name))
            saved_count += 1

    print(f"Завантажено {saved_count} фото для {student_name} в {student_dir}")
    return redirect(url_for('edit_class', class_name=class_name))


@app.route('/train_model/<class_name>', methods=['POST'])
def train_model(class_name):
    try:

        result = subprocess.run(["python", "custom_trainer.py"], capture_output=True, text=True)

        if result.returncode == 0:
            print("Навчання успішно завершено!")
            print(result.stdout)
        else:
            print(f"Помилка під час навчання: {result.stderr}")

    except Exception as e:
        print(f"Не вдалося запустити custom_trainer.py: {e}")


    return redirect(url_for('edit_class', class_name=class_name))

@app.route('/start_search')
def start_search():
    global is_searching, search_start_time, selected_face_index
    is_searching = True
    search_start_time = time.time()
    selected_face_index = -1
    print("Сигнал на вибір учня отримано!")
    return "OK"

@app.route('/get_present_students')
def get_present_students():
    return {"present": current_present_students}


# @app.route('/grant_immunity', methods=['POST'])
# def grant_immunity():
#     global is_immunity_searching, immunity_search_start, immune_students
#
#     data = request.get_json()
#     present_on_lesson = data.get('present_students', [])
#
#     # Вибираємо тільки з тих, кого ви відмітили і хто ще не має імунітету
#     candidates = [s for s in present_on_lesson if s not in immune_students]
#
#     if candidates:
#         winner = random.choice(candidates)
#         immune_students.append(winner)
#
#         # Вмикаємо режим анімації на відео на 3 секунди
#         is_immunity_searching = True
#         immunity_search_start = time.time()
#
#         return jsonify({"status": "success", "student": winner})
#
#     return jsonify({"status": "error", "message": "Немає кого обирати"})

def select_student_logic(candidates, student_counts):
    """
    candidates: список імен присутніх учнів без імунітету
    student_counts: словник {ім'я: кількість_виходів}
    """
    if not candidates:
        return None

    # Обчисл. ваги. Чим менше виходів, тим більша вага.

    weights = []
    for name in candidates:
        count = student_counts.get(name, 0)
        # Формула: шанс = 1 / (кількість_виходів + 1)
        # Можна змінити на (1 / (count + 1))**2
        weight = 1.0 / (count + 1)
        weights.append(weight)

    #  Випадковий вибір з урахуванням ваг

    winner = random.choices(candidates, weights=weights, k=1)[0]
    return winner


@app.route('/call_random_student', methods=['POST'])
def call_random_student():
    global student_counts, immune_students

    data = request.get_json()
    present_students = data.get('present_students', [])

    # Фільтрує тих, хто присутній і НЕ має імунітету
    candidates = [s for s in present_students if s not in immune_students]

    if not candidates:
        return jsonify({"status": "error", "message": "Немає доступних учнів"})

    chosen_one = select_student_logic(candidates, student_counts)


    global is_immunity_searching, immunity_search_start, immunity_winner
    immunity_winner = chosen_one  # Використовуємо ту ж змінну для анімації
    is_immunity_searching = True
    immunity_search_start = time.time()

    return jsonify({"status": "success", "student": chosen_one})


@app.route('/toggle_immunity', methods=['POST'])
def toggle_immunity():
    global immune_students
    data = request.get_json()
    student_name = data.get('student')

    if student_name in immune_students:
        immune_students.remove(student_name)
        status = "removed"
    else:
        immune_students.append(student_name)
        status = "added"

    return jsonify({"status": "success", "action": status})

@app.route('/reset_immunity', methods=['POST'])
def reset_immunity():
    global immune_students
    immune_students = []  # Повністю очищаємо список
    return jsonify({"status": "success", "message": "Імунітети скинуто"})

@app.route('/reset_full_table', methods=['POST'])
def reset_full_table():
    global immune_students, student_counts
    immune_students = []
    # Обнуляє лічильники для всіх учнів
    student_counts = {name: 0 for name in student_counts}
    return jsonify({"status": "success"})


@app.route('/export_journal')
def export_journal():
    # 1. Отримуємо актуальний список учнів (з папок)
    students_list = get_current_students()

    #  Формує дані для таблиці
    data = []
    for name in students_list:

        count = student_counts.get(name, 0)
        # Перевіряє наявність у списку імунітетів
        has_immunity = "Так" if name in immune_students else "Ні"

        data.append({
            "Учень": name,
            "Кількість виходів": count,
            "Імунітет": has_immunity
        })

    # зберігає в Excel
    df = pd.DataFrame(data)

    file_path = "MyClassAI_Journal.xlsx"
    df.to_excel(file_path, index=False, engine='openpyxl')

    #  Відправляє файл користувачу
    return send_file(file_path, as_attachment=True)

@app.route('/get_counts')
def get_counts():
    global student_counts
    return jsonify(student_counts)

if __name__ == '__main__':
    app.run(debug=True)
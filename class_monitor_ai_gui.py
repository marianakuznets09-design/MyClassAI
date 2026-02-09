import tkinter as tk
from tkinter import messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import random  # Необхідний для зваженого вибору

# Глобальні Змінні (Ініціалізуються в class_monitor_ai.py)

# у головному файлі (class_monitor_ai.py)
cap = None
recognizing_active = False
face_cascade = None
current_frame = None
last_choice_time = 0
chosen_class_index = None
selection_animation = {'active': False, 'start_time': 0}
immunity_animation = {'active': False, 'start_time': 0, 'index': None}

# Поріг впевненості для відображення "Невідомо"



class ClassMonitorGUI:
    def __init__(self, root, model, db_manager, known_face_names):
        self.root = root
        self.model = model
        self.db_manager = db_manager
        self.known_face_names = known_face_names

        self.IMAGE_SIZE = 96  # Розмір вхідного зображення для моделі
        self.CONFIDENCE_THRESHOLD = 0.65

        # Ініціалізація змінних стану
        self.current_frame_student_indices = []
        self.current_frame_student_boxes = {}
        self.chosen_class_index = None
        self.selection_animation = {'active': False, 'start_time': 0}
        self.auto_select_active = False  # 🟢 Ініціалізація для кнопки автовибору

        self.font_path = "arial.ttf"
        self.font_size = 20



        try:
            # Завантажуємо шрифт
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            print(f" Помилка завантаження шрифту '{self.font_path}'. Використовується шрифт за замовчуванням.")
            self.font = ImageFont.load_default()


            self.setup_ui()





        root.title("Class Monitor AI")
        root.geometry("1200x800")
        root.configure(bg='grey10')

        # Налаштування grid для головного вікна
        root.grid_columnconfigure(0, weight=3)  # Відео
        root.grid_columnconfigure(1, weight=1)  # Панель керування
        root.grid_rowconfigure(0, weight=1)

        # Ліва панель: Відео
        video_frame = tk.Frame(root, bg='black', bd=5, relief="raised")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.status_label = tk.Label(video_frame, text="Натисніть 'Увімкнути Камеру'", fg="yellow", bg="black",
                                     font=("Arial", 14))
        self.status_label.grid(row=1, column=0, sticky="ew")

        # Права панель: Керування та Статистика
        self.control_frame = tk.Frame(root, bg='grey10', bd=5, relief="ridge")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Секція: Керування Системою (Виправлено на Grid)
        tk.Label(self.control_frame, text="Керування Системою", fg="cyan", bg="grey10",
                 font=("Arial", 16, "bold")).pack(pady=10)

        # 1. ГОЛОВНІ КНОПКИ
        self.btn_toggle_camera = tk.Button(self.control_frame, text="Увімкнути Камеру", command=self.toggle_recognition,
                                           bg="green", fg="white", font=("Arial", 14, "bold"), height=2)
        self.btn_toggle_camera.pack(fill=tk.X, padx=5, pady=5)

        self.btn_stop_camera = tk.Button(self.control_frame, text="Вимкнути Камеру",
                                         command=lambda: self.toggle_recognition(False),
                                         bg="red", fg="white", font=("Arial", 14, "bold"), height=2)
        self.btn_stop_camera.pack(fill=tk.X, padx=5, pady=5)

        # 2. ДОПОМІЖНІ КНОПКИ
        button_grid_frame = tk.Frame(self.control_frame, bg="grey10")
        button_grid_frame.pack(fill=tk.X, padx=5, pady=5)
        button_grid_frame.grid_columnconfigure(0, weight=1)
        button_grid_frame.grid_columnconfigure(1, weight=1)

        def create_grid_button(text, command, color, row, col):
            btn = tk.Button(button_grid_frame, text=text, command=command, bg=color, fg="white",
                            font=("Arial", 10, "bold"), height=2)
            btn.grid(row=row, column=col, sticky="nsew", padx=3, pady=1)
            return btn

        #  Сітка Кнопок

        self.btn_select_board = create_grid_button("Вибрати до дошки (S)", self.select_student_for_board, "blue", 0, 0)
        # self.btn_random_select = create_grid_button("Випадковий вибір", self.select_random_student, "dark blue", 0, 1)


        self.btn_grant_immunity = create_grid_button("Надати Імунітет (I)", lambda: self.set_immunity_status(True),
                                                     "purple", 1, 0)
        self.btn_remove_immunity = create_grid_button("Зняти Імунітет (R)", lambda: self.set_immunity_status(False),
                                                      "brown", 1, 1)


        self.btn_auto_select = create_grid_button("Увімк./Вимк. Автовибір", self.toggle_auto_select, "dim grey", 2, 0)
        self.btn_toggle_report = create_grid_button("Увімк./Вимк. Звіт (T)", self.toggle_detailed_stats, "grey", 2, 1)

        # Кнопки Скидання
        self.btn_reset_counts = create_grid_button("Скинути Лічильники", self.reset_output_counts, "orange", 3, 0)
        self.btn_reset_all = create_grid_button("Скинути ВСЕ", self.reset_all_data, "black", 3, 1)

        # Зона для поточної статистики
        tk.Label(self.control_frame, text="Поточна Статистика", fg="white", bg="grey10",
                 font=("Arial", 13, "bold")).pack(pady=10)
        self.stats_label = tk.Label(self.control_frame, text="Оновлення...", fg="light grey", bg="grey15",
                                    font=("Courier New", 12), justify=tk.LEFT, anchor='w', height=3)
        self.stats_label.pack(fill=tk.X, padx=5, pady=5)

        # --- Зона для детальної статистики
        self.detailed_stats_frame = tk.Frame(self.control_frame, bg="grey10", bd=2, relief="groove")
        self.detailed_stats_frame.pack(pady=10, padx=5, fill=tk.BOTH, expand=True)

        tk.Label(self.detailed_stats_frame, text="Детальний Звіт", fg="cyan", bg="grey10",
                 font=("Arial", 13, "bold")).pack(pady=5)

        text_wrapper = tk.Frame(self.detailed_stats_frame, bg="grey15")
        text_wrapper.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_wrapper.grid_rowconfigure(0, weight=1)
        text_wrapper.grid_columnconfigure(0, weight=1)

        # 1. Створення Scrollbar
        scrollbar = tk.Scrollbar(text_wrapper)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 2. Створення Text Widget
        self.detailed_stats_text = tk.Text(text_wrapper, bg="grey15", fg="white", font=("Courier New", 10),
                                           height=10, wrap="word", relief="flat", insertbackground="white",
                                           yscrollcommand=scrollbar.set)
        self.detailed_stats_text.grid(row=0, column=0, sticky="nsew")

        # 3. Зв'язок Scrollbar -> Text
        scrollbar.config(command=self.detailed_stats_text.yview)

        self.detailed_stats_text.config(state=tk.DISABLED)
        self.detailed_stats_visible = False  # За замовчуванням приховано

        # Обробка закриття вікна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Ініціалізує та розміщує всі елементи графічного інтерфейсу Tkinter."""

        # 1. Загальні налаштування вікна
        self.root.title("Class Monitor AI")
        self.root.geometry("1200x800")
        self.root.configure(bg='grey10')

        # 2. Налаштування grid для головного вікна
        self.root.grid_columnconfigure(0, weight=3)  # Відео
        self.root.grid_columnconfigure(1, weight=1)  # Панель керування
        self.root.grid_rowconfigure(0, weight=1)

        # 3.Ліва панель: Відео (та весь код відеопанелі)
        video_frame = tk.Frame(self.root, bg='black', bd=5, relief="raised")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.status_label = tk.Label(video_frame, text="Натисніть 'Увімкнути Камеру'", fg="yellow", bg="black",
                                     font=("Arial", 14))
        self.status_label.grid(row=1, column=0, sticky="ew")

        # 4. Права панель: Керування та Статистика
        self.control_frame = tk.Frame(self.root, bg='grey10', bd=5, relief="ridge")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)



        # 5. Обробка закриття вікна (Залишаємо)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.slow_stats_update()

    def start_camera(self):
        """Ініціалізує камеру. Запускається один раз при старті."""

        self.toggle_recognition(True)

    def start_recognition(self):
        """Метод-заглушка для старту, якщо логіка вже в toggle_recognition."""

        pass

    #                   МЕТОДИ ОБРОБКИ


    #1.
    def toggle_recognition(self, state=None):
        global cap, recognizing_active

        # Увімкнення
        if state is True or (state is None and not recognizing_active):
            cap = cv2.VideoCapture(0)
            if cap and cap.isOpened():
                recognizing_active = True
                threading.Thread(target=self.video_stream, daemon=True).start()
                self.status_label.config(text="Камера УВІМКНЕНА. Розпізнавання активне.", fg="green")
            else:
                self.status_label.config(text="Помилка: Не вдалося відкрити камеру.", fg="red")

        # Вимкнення
        elif state is False or (state is None and recognizing_active):
            recognizing_active = False
            if cap:
                cap.release()
                # Очистити зображення у віджеті, щоб прибрати "завислий" кадр
                self.video_label.config(image='')
                # Видалити посилання, щоб звільнити пам'ять (запобігання memory leaks)
                self.video_label.imgtk = None
            self.status_label.config(text="Камера ВИМКНЕНА.", fg="red")

    #2.
    def select_student_for_board(self):
        """Вибирає учня до дошки серед присутніх, зважено на статистику виходів."""

        # Отримуємо індекси присутніх (без імунітету)
        available_indices = [
            idx for idx in self.current_frame_student_indices
            if not self.check_immunity_status(idx)
        ]

        if not available_indices:
            messagebox.showinfo("Вибір", "Нікого не видно в кадрі або всі присутні імунні.")
            return


        # 1. Розрахунок Ваг
        weights = [
            1.0 / (self.db_manager.get_output_count(self.known_face_names[idx]) + 1)  # Звернення до DB
            for idx in available_indices
        ]

        # 2. Зважений Випадковий Вибір
        chosen_index = random.choices(available_indices, weights=weights, k=1)[0]


        self.chosen_class_index = chosen_index
        self.selection_animation['active'] = True
        self.selection_animation['start_time'] = time.time()

        name = self.known_face_names[chosen_index]

        self.db_manager.increment_output_count(name)
        messagebox.showinfo("Вибір", f"До дошки викликається (присутній, зважено): {name}!")

    #  3.
    def select_random_student(self):
        """Вибирає випадкового учня з усієї бази, зважено на статистику виходів."""

        # Отримуємо всі індекси, які НЕ мають імунітету (з усієї бази)
        available_indices = [
            idx for idx in range(len(self.known_face_names))
            if not self.check_immunity_status(idx)
        ]

        if not available_indices:
            messagebox.showinfo("Вибір", "Всі учні мають імунітет, або база порожня.")
            return


        # 1. Розрахунок Ваг
        weights = [
            1.0 / (self.db_manager.get_output_count(self.known_face_names[idx]) + 1)
            for idx in available_indices
        ]

        # 2. Зважений Випадковий Вибір
        chosen_index = random.choices(available_indices, weights=weights, k=1)[0]


        self.chosen_class_index = chosen_index
        self.selection_animation['active'] = True
        self.selection_animation['start_time'] = time.time()

        name = self.known_face_names[chosen_index]

        self.db_manager.increment_output_count(name)

        messagebox.showinfo("Вибір", f"До дошки викликається (випадково, зважено): {name}!")

    #  4.  toggle_auto_select
    def toggle_auto_select(self):
        """Перемикає стан автоматичного вибору учнів до дошки."""

        self.auto_select_active = not self.auto_select_active

        if self.auto_select_active:
            self.btn_auto_select.config(text="Автовибір УВІМКНЕНО", bg="gold3")

            # threading.Thread(target=self.auto_selection_loop, daemon=True).start()
        else:
            self.btn_auto_select.config(text="Увімк./Вимк. Автовибір", bg="dim grey")

        messagebox.showinfo("Автовибір",
                            f"Автоматичний вибір тепер: {'УВІМКНЕНО' if self.auto_select_active else 'ВИМКНЕНО'}")

    # 5. Інші методи керування
    def set_immunity_status(self, grant=True):
        #  логіка імунітету
        pass

    def toggle_detailed_stats(self):
        self.detailed_stats_visible = not self.detailed_stats_visible
        self.update_detailed_stats()

    def reset_output_counts(self):
        #  логіка скидання лічильників
        if messagebox.askyesno("Скидання", "Ви впевнені, що хочете скинути лічильники виходів для всіх учнів?"):
            self.db_manager.reset_all_output_counts()
            messagebox.showinfo("Скидання", "Лічильники успішно скинуто.")
            self.update_current_stats()

    def reset_all_data(self):

        pass

    def check_immunity_status(self, index):
        # Припускаємо, що це береться з db_manager
        name = self.known_face_names[index]
        return self.db_manager.get_immunity_status(name)

    def get_output_count(self, name):
        # Ця функція є критичною для зваженого вибору
        return self.db_manager.get_output_count(name)


    #                   МЕТОД ОБРОБКИ ВІДЕО


    def video_stream(self):
        """Основний потік для обробки відео з камери."""
        # 1. Очищення оголошень global та DEBUG-друку

        global cap, recognizing_active, current_frame, last_choice_time, \
            chosen_class_index, selection_animation, immunity_animation


        if not hasattr(self, 'face_cascade') or self.face_cascade is None:
            print("❌ face_cascade не було передано як атрибут. Вихід з потоку.")
            global recognizing_active
            recognizing_active = False
            return


        try:
            # 2. Перевірка камери
            if cap is None or not cap.isOpened():
                print("Камера не була успішно ініціалізована. Вихід.")
                return

            while recognizing_active:  # cap.isOpened() перевіряється в toggle_recognition

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                current_frame = frame.copy()
                assigned_names_in_frame = set()

                # Масштабування для швидшого розпізнавання
                frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                # cvtColor використовує BGR (OpenCV) -> RGB (Keras)
                frame_rgb_small = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                # 3. Знаходження обличчя за допомогою Haar Cascade
                # використовуємо self.face_cascade
                face_locations = self.face_cascade.detectMultiScale(
                    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                current_frame_student_indices = []
                current_frame_student_boxes = {}

                if face_locations is not None and len(face_locations) > 0:

                    # Обробка кожного знайденого обличчя
                    for (x, y, w, h) in face_locations:

                        # Підготовка обличчя для моделі Keras (ROI з масштабованого кадру)
                        face_img = frame_rgb_small[y:y + h, x:x + w]
                        if face_img.size == 0:
                            continue

                        # Масштабування та нормалізація для моделі
                        face_img = cv2.resize(face_img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                        face_img_normalized = np.expand_dims(face_img.astype('float32') / 255.0, axis=0)


                        predictions = self.model.predict(face_img_normalized, verbose=0)

                        # Отримання класу та впевненості
                        predicted_class_index = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class_index]

                        # Масштабування координат назад до повного кадру
                        left = x * 2
                        top = y * 2
                        right = (x + w) * 2
                        bottom = (y + h) * 2

                        box_color = (0, 255, 0)  # Колір за замовчуванням (зелений)
                        name = "Невідомо"
                        confidence_percent = f" ({confidence * 100:.1f}%)"

                        # 4. Логіка Розпізнавання та Статусів
                        if confidence >= self.CONFIDENCE_THRESHOLD:
                            current_index = predicted_class_index

                            if not self.known_face_names:
                                name = "Невідомо (База порожня)"
                            elif current_index < len(self.known_face_names):
                                potential_name = self.known_face_names[current_index]

                                # ПЕРЕВІРКА, чи використовується ім'я вже в цьому кадрі
                                if potential_name in assigned_names_in_frame:
                                    # Якщо ім'я вже використовується

                                    name = "Невідомо (Дублювання)"
                                    box_color = (0, 165, 255)  # Помаранчевий
                                else:
                                    # Ім'я унікальне, призначаємо його та додаємо до використаних
                                    name = potential_name
                                    assigned_names_in_frame.add(name)  # 🟢 ДОДАЄМО Унікальне ім'я до набору

                            else:
                                name = "Невідомо (Індекс не знайдено)"

                            # 4.1. Логіка ІМУНІТЕТУ
                            if name not in ["Невідомо (Дублювання)", "Невідомо (База порожня)",
                                            "Невідомо (Індекс не знайдено)"]:
                                if self.db_manager.get_immunity_status(name):
                                    box_color = (180, 105, 255)  # Фіолетовий
                                    name += " (ІМУНІТЕТ)"

                                # 4.2. Логіка ВИКЛИКУ ДО ДОШКИ
                                if current_index == chosen_class_index and not selection_animation['active']:
                                    box_color = (0, 0, 255)  # Червоний
                                    name += " (ДО ДОШКИ!)"

                                current_frame_student_indices.append(current_index)
                                current_frame_student_boxes[current_index] = (left, top, right, bottom)

                            # name += confidence_percent

                        else:
                            # Низька впевненість
                            box_color = (0, 165, 255)
                            name = "Невідомо" #+ confidence_percent

                        # 5.

                        # 5.1. Малюємо прямокутник
                        cv2.rectangle(current_frame, (left, top), (right, bottom), box_color, 2)

                        # 5.2. Обробка тексту Юнікоду за допомогою PIL

                        img_pil = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)

                        :
                        R, G, B = box_color[2], box_color[1], box_color[0]
                        text_color = (R, G, B)

                        # Розташування тексту (вище прямокутника)
                        draw.text((left, top - 25), name, font=self.font, fill=text_color)

                        # 5.3. Перетворює кадр PIL назад на OpenCV (BGR)
                        current_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # 6. Оновлення статистики
                self.current_frame_student_indices = current_frame_student_indices
                self.current_frame_student_boxes = current_frame_student_boxes
                self.update_current_stats()

                # 7. Відображення кадру у Tkinter
                img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(img)


                imgtk = ImageTk.PhotoImage(image=img)


                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                time.sleep(0.03)

        except Exception as e:
            # Обробка критичної помилки потоку
            print(f"Критична помилка у потоці відео: {e}")
            recognizing_active = False  # Зупиняємо цикл

        finally:
            if cap and cap.isOpened():
                cap.release()
            print("Камера вимкнена.")

    # ---------------------
    #МЕТОДИ ОНОВЛЕННЯ GUI


    def update_current_stats(self):
        """Оновлює поточну статистику учнів."""

        total_students = len(self.known_face_names)
        present_count = len(set(self.current_frame_student_indices))

        # Отримання даних з бази даних
        immune_count = self.db_manager.get_total_immune_count()

        stats_text = (
            f"Всього учнів: {total_students}\n"
            f"Присутні (в кадрі): {present_count}\n"
            f"Імунітет: {immune_count}"
        )
        self.stats_label.config(text=stats_text)

        # Оновлення детального звіту (якщо активний)
        # if self.detailed_stats_visible:
        #     self.update_detailed_stats()

    def slow_stats_update(self):
        """Оновлює детальний звіт рідше, щоб уникнути затримок."""
        if self.detailed_stats_visible:
            self.update_detailed_stats()

        # Викликати цю функцію знову через 2000 мс (2 секунди)
        self.root.after(2000, self.slow_stats_update)

    def update_detailed_stats(self):
        """Оновлює детальний звіт (зважено на присутність)."""
        if not self.detailed_stats_visible:
            self.detailed_stats_text.config(state=tk.DISABLED)
            return

        self.detailed_stats_text.config(state=tk.NORMAL)
        self.detailed_stats_text.delete(1.0, tk.END)

        detailed_data = self.db_manager.get_all_student_data()
        report = "Ім'я | Виходи | Імунітет | Присутність\n"
        report += "---------------------------------------\n"

        present_indices = set(self.current_frame_student_indices)

        for data in detailed_data:
            idx = self.known_face_names.index(data['name']) if data['name'] in self.known_face_names else -1

            # Статус присутності
            presence = "✅" if idx != -1 and idx in present_indices else "❌"

            report += (
                f"{data['name']:<18} | "
                f"{data['output_count']:<6} | "
                f"{'🛡️' if data['is_immune'] else ' ':<8} | "
                f"{presence}\n"
            )

        self.detailed_stats_text.insert(tk.END, report)
        self.detailed_stats_text.config(state=tk.DISABLED)

    # 7. Обробка закриття вікна
    def on_closing(self):
        global recognizing_active, cap
        recognizing_active = False
        if cap and cap.isOpened():
            cap.release()
        self.root.destroy()
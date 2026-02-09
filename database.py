import sqlite3

DATABASE_NAME = 'new_class_data.db'

def setup_db():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    #  таблиця класів
    cursor.execute('''CREATE TABLE IF NOT EXISTS Classes 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE)''')
    # таблиця учнів зі зв'язком до класу
    cursor.execute('''CREATE TABLE IF NOT EXISTS Students 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       name TEXT, 
                       times_chosen INTEGER DEFAULT 0, 
                       class_id INTEGER,
                       FOREIGN KEY(class_id) REFERENCES Classes(id) ON DELETE CASCADE)''')
    conn.commit()
    conn.close()

def add_class_to_db(name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO Classes (name) VALUES (?)", (name,))
        conn.commit()
    except: pass # Якщо клас вже існує
    conn.close()

def get_all_classes():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM Classes")
    classes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return classes

def delete_class_from_db(name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM Classes WHERE name = ?", (name,))
    conn.commit()
    conn.close()


def add_student_to_db(name, class_name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM Classes WHERE name = ?", (class_name,))
    class_id = cursor.fetchone()[0]

    # Додається учня, прив'язуючи його до цього ID
    cursor.execute("INSERT INTO Students (name, class_id) VALUES (?, ?)", (name, class_id))
    conn.commit()
    conn.close()


def get_students_by_class(class_name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''SELECT Students.name FROM Students 
                      JOIN Classes ON Students.class_id = Classes.id 
                      WHERE Classes.name = ?''', (class_name,))
    students = [row[0] for row in cursor.fetchall()]
    conn.close()
    return students
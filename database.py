import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "recommender.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            interests TEXT DEFAULT '',
            career_goal TEXT DEFAULT '',
            skill_level TEXT DEFAULT 'beginner',
            weekly_hours INTEGER DEFAULT 5
        )
    """)
    
    # Create saved_courses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_name TEXT NOT NULL,
            difficulty TEXT,
            rating TEXT,
            url TEXT,
            pacing TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    # Create completed_interviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS completed_interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            experience_level TEXT NOT NULL,
            interview_type TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            num_questions INTEGER NOT NULL,
            history_json TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username.strip(), hashed_password)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        user_id = None
    finally:
        conn.close()
    return user_id

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user["password"], password):
        return user
    return None

def update_user_preferences(user_id, interests, career_goal, skill_level, weekly_hours):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users 
        SET interests = ?, career_goal = ?, skill_level = ?, weekly_hours = ?
        WHERE id = ?
    """, (interests.strip(), career_goal.strip(), skill_level.strip(), int(weekly_hours), user_id))
    conn.commit()
    conn.close()

def get_user_preferences(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT interests, career_goal, skill_level, weekly_hours FROM users WHERE id = ?", (user_id,))
    prefs = cursor.fetchone()
    conn.close()
    return prefs

def save_course(user_id, course_name, difficulty, rating, url, pacing):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check if already bookmarked
    cursor.execute("""
        SELECT id FROM saved_courses 
        WHERE user_id = ? AND course_name = ?
    """, (user_id, course_name))
    if cursor.fetchone() is not None:
        conn.close()
        return False # Already exists
        
    cursor.execute("""
        INSERT INTO saved_courses (user_id, course_name, difficulty, rating, url, pacing)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, course_name, difficulty, rating, url, pacing))
    conn.commit()
    conn.close()
    return True

def get_saved_courses(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, course_name, difficulty, rating, url, pacing FROM saved_courses WHERE user_id = ?", (user_id,))
    courses = cursor.fetchall()
    conn.close()
    return courses

def delete_saved_course(course_id, user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_courses WHERE id = ? AND user_id = ?", (course_id, user_id))
    conn.commit()
    conn.close()

def save_completed_interview(user_id, role, experience_level, interview_type, difficulty, num_questions, history_json):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO completed_interviews (user_id, role, experience_level, interview_type, difficulty, num_questions, history_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, role, experience_level, interview_type, difficulty, int(num_questions), history_json))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_completed_interviews(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, role, experience_level, interview_type, difficulty, num_questions, history_json, timestamp 
        FROM completed_interviews 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

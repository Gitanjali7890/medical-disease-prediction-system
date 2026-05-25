# database.py - User Authentication Database
import sqlite3
import hashlib
from datetime import datetime
import os

DB_PATH = 'models/users.db'

def init_db():
    """Initialize database tables"""
    os.makedirs('models', exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symptoms TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            specialists TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized!")

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password):
    """Create new user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed)
        )
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {'success': True, 'user_id': user_id}
    except sqlite3.IntegrityError as e:
        conn.close()
        if 'username' in str(e):
            return {'success': False, 'error': 'Username already exists'}
        elif 'email' in str(e):
            return {'success': False, 'error': 'Email already registered'}
        return {'success': False, 'error': 'Registration failed'}

def authenticate_user(username, password):
    """Authenticate user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    hashed = hash_password(password)
    cursor.execute(
        "SELECT id, username, email FROM users WHERE username = ? AND password = ?",
        (username, hashed)
    )
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {'success': True, 'user_id': user[0], 'username': user[1], 'email': user[2]}
    return {'success': False, 'error': 'Invalid username or password'}

def save_prediction(user_id, symptoms, disease, confidence, severity, specialists):
    """Save prediction to history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (user_id, symptoms, predicted_disease, confidence, severity, specialists)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, symptoms, disease, confidence, severity, specialists))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_user_predictions(user_id, limit=20):
    """Get user's prediction history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, symptoms, predicted_disease, confidence, severity, created_at
        FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
    ''', (user_id, limit))
    
    results = cursor.fetchall()
    conn.close()
    
    predictions = []
    for row in results:
        predictions.append({
            'id': row[0],
            'symptoms': row[1],
            'disease': row[2],
            'confidence': row[3],
            'severity': row[4],
            'date': row[5]
        })
    return predictions

# Initialize database
init_db()
import sqlite3
from datetime import datetime
import bcrypt
from typing import Optional, List, Dict

class Database:
    def __init__(self, db_path: str = "skin_disease_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, email: str, password: str) -> bool:
        """Create a new user with hashed password"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Hash the password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email, password_hash)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def verify_user(self, email: str, password: str) -> Optional[int]:
        """Verify user credentials and return user ID if valid"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result[1]):
            return result[0]
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user information by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, email, created_at FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'email': result[1],
                'created_at': result[2]
            }
        return None
    
    def save_prediction(self, user_id: int, image_filename: str, predicted_class: str, confidence: float):
        """Save a prediction to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO predictions (user_id, image_filename, predicted_class, confidence) VALUES (?, ?, ?, ?)",
            (user_id, image_filename, predicted_class, confidence)
        )
        conn.commit()
        conn.close()
    
    def get_user_predictions(self, user_id: int) -> List[Dict]:
        """Get all predictions for a specific user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, image_filename, predicted_class, confidence, created_at FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        results = cursor.fetchall()
        conn.close()
        
        predictions = []
        for result in results:
            predictions.append({
                'id': result[0],
                'image_filename': result[1],
                'predicted_class': result[2],
                'confidence': result[3],
                'created_at': result[4]
            })
        
        return predictions
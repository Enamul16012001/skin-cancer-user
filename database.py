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
        
        # Check if is_admin column exists, if not add it
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_admin' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")
        
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
        
        # Create default admin user if doesn't exist
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            # Create default admin with password "admin123"
            admin_password_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
                ("admin@skindisease.com", admin_password_hash, True)
            )
        
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
        
        cursor.execute("SELECT id, email, is_admin, created_at FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'email': result[1],
                'is_admin': result[2],
                'created_at': result[3]
            }
        return None
    
    def delete_user_account(self, user_id: int, password: str) -> bool:
        """Delete user account after password verification"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Verify password first
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        if not result or not bcrypt.checkpw(password.encode('utf-8'), result[0]):
            conn.close()
            return False
        
        # Delete user's predictions first (foreign key constraint)
        cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        
        # Delete user account
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        return True
    
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
    
    def delete_prediction(self, prediction_id: int, user_id: int) -> bool:
        """Delete a specific prediction (only if it belongs to the user)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id)
        )
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    # Admin functions
    def get_all_users(self) -> List[Dict]:
        """Get all users (admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, email, is_admin, created_at, "
            "(SELECT COUNT(*) FROM predictions WHERE user_id = users.id) as prediction_count "
            "FROM users ORDER BY created_at DESC"
        )
        results = cursor.fetchall()
        conn.close()
        
        users = []
        for result in results:
            users.append({
                'id': result[0],
                'email': result[1],
                'is_admin': result[2],
                'created_at': result[3],
                'prediction_count': result[4]
            })
        
        return users
    
    def get_all_predictions(self) -> List[Dict]:
        """Get all predictions with user info (admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT p.id, p.image_filename, p.predicted_class, p.confidence, p.created_at, u.email "
            "FROM predictions p JOIN users u ON p.user_id = u.id ORDER BY p.created_at DESC"
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
                'created_at': result[4],
                'user_email': result[5]
            })
        
        return predictions
    
    def admin_delete_user(self, user_id: int) -> bool:
        """Admin delete user (no password verification needed)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Don't allow deleting admin users (use 1 instead of TRUE for SQLite compatibility)
        cursor.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if result and result[0] == 1:  # is_admin is 1 (True)
            conn.close()
            return False
        
        # Delete user's predictions first
        cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        
        # Delete user account
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    def admin_delete_prediction(self, prediction_id: int) -> bool:
        """Admin delete prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
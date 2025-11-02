import sqlite_utils
from dotenv import load_dotenv
import os

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "./todo.db")

def get_db():
    if not DB_PATH:
        raise ValueError("DB_PATH is not set")
    db = sqlite_utils.Database(DB_PATH)
    db["todos"].create(
        {
            "id": int,
            "title": str,
            "description": str,
            "completed": bool,
            "created_at": str,
        },
        pk="id", if_not_exists=True
    )
    return db

def init_db():
    get_db()
    
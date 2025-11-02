# Seed the database with initial data
from database import init_db, get_db
from datetime import datetime

def seed_db():
    init_db()
    db = get_db()
    table = db["todos"]

    # Sample todo items
    todos = [
        {
            "title": "Buy groceries",
            "description": "Milk, Bread, Eggs, Butter",
            "completed": False,
            "created_at": datetime.now().isoformat(),
        },
        {
            "title": "Read a book",
            "description": "Finish reading 'The Great Gatsby'",
            "completed": False,
            "created_at": datetime.now().isoformat(),
        },
        {
            "title": "Exercise",
            "description": "Go for a 30-minute run",
            "completed": False,
            "created_at": datetime.now().isoformat(),
        },
    ]

    for todo in todos:
        table.insert(todo)

if __name__ == "__main__":
    seed_db()
    print("Database seeded with initial todo items.")
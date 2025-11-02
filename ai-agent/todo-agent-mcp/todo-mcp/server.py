from mcp.server.fastmcp import FastMCP
from database import init_db, get_db
from models import Todo, TodoCreate, TodoUpdate, TodoList
from datetime import datetime
from typing import Dict, Any
import json

# initialize the database
init_db()

# get the database instance
db = get_db()
table = db["todos"]

# create a mcp server instance
app = FastMCP("my-todo-mcp")

"""
Tools:
- add_todo: Add a new todo item
- update_todo: Update a todo item
- delete_todo: Delete a todo item
- complete_todo: Mark a todo item as completed
"""

@app.tool()
def add_todo(todo: TodoCreate) -> Dict[str, Any]:
    """Add a new todo item"""
    try:
        now = datetime.now().isoformat()
        todo_data = todo.model_dump() | {"created_at": now}
        table.insert(todo_data)
        return {"message": "Todo added successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def update_todo(id: int, todo: TodoUpdate) -> Dict[str, Any]:
    """Update a todo item"""
    try:
        existing_todo = table.get(id)
        if not existing_todo:
            return {"error": "Todo item not found"}
        # Exclude None values to avoid overwriting existing fields with null values
        updated_data = todo.model_dump(exclude_none=True)
        table.update(
            id,
            updated_data
        )
        return {"message": "Todo updated successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def delete_todo(id: int) -> Dict[str, Any]:
    """Delete a todo item"""
    try:
        table.delete(id)
        return {"message": "Todo deleted successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def complete_todo(id: int) -> Dict[str, Any]:
    """Mark a todo item as completed"""
    try:
        todo = table.get(id)
        if not todo:
            return {"error": "Todo item not found"}
        table.update(
            id,
            {
                "completed": True,
            }
        )
        return {"message": "Todo marked as completed"}
    except Exception as e:
        return {"error": str(e)}
    

# Helper function to get todos
def _get_todos(completed: bool | None = None) -> TodoList | Dict[str, Any]:
    try:
        if completed is not None:
            rows = [row for row in table.rows if row["completed"] == completed]
        else:
            rows = list(table.rows)
        todos = [Todo(**row) for row in rows]
        return TodoList(todos=todos)
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def list_all_todos():
    """List all todo items"""
    try:
        todos = _get_todos()
        return json.dumps(todos.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def get_todo(id: int):
    """Get a todo item by ID"""
    try:
        todo = table.get(id)
        if not todo:
            return {"error": "Todo item not found"}
        return {"todo": todo}
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def list_completed_todos():
    """List all completed todo items"""
    try:
        todos = _get_todos(completed=True)
        return json.dumps(todos.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def list_pending_todos():
    """List all pending todo items"""
    try:
        todos = _get_todos(completed=False)
        return json.dumps(todos.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:
        return {"error": str(e)}

"""
Prompts
- daily-summary: Generate a daily summary of todos
- action-plan: Generate an action plan for the next day
"""

@app.prompt("daily-summary")
def daily_summary_prompt(pending: str, done: str):
    """Generate a daily summary of todos"""
    return f"""
        You are a productivity assistant.

        Pending tasks:
        {pending}

        Tasks completed today:
        {done}

        Provide a motivating summary in 3 lines:
        1. What has been accomplished
        2. What remains to be done
        3. Today's advice for staying productive.
    """

@app.prompt("action-plan")
def action_plan_prompt(todos: str):
    """Generate an action plan for the next day"""
    return f"""
        You are a productivity assistant.

        Tasks to do:
        {todos}
        Create a concise action plan, prioritizing tasks and suggesting time blocks for each.
    """

if __name__ == "__main__":
    print("Todo MCP Server running")
    print("Connect it in Claude, Cursor, VS Code, Langchain agent, ...")
    app.run()
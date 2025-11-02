from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime

class Todo(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False
    created_at: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class TodoCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=500, description="Title of the todo item")
    description: Optional[str] = Field(default=None, description="Description of the todo item")
    completed: bool = False

    @field_validator('title')
    def title_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Title must not be empty')
        return v
    
    @field_validator('description')
    def description_default(cls, v):
        return v or ""

class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    completed: Optional[bool] = None

class TodoList(BaseModel):
    todos: List[Todo] = Field(default_factory=list)

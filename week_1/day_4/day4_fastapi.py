import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Todo(BaseModel):
    title: str
    completed: bool


@app.get("/")
def home():
    return {"message": "Day 4 API is working"}


@app.get("/hello")
def hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/todo/{todo_id}")
def get_todo(todo_id: int):
    return {"todo_id": todo_id}


@app.post("/todo")
def create_todo(todo: Todo):
    return {
        "message": "Todo created",
        "todo": todo
    }


@app.get("/search")
def search_todos(completed: bool = False):
    return {
        "completed": completed,
        "message": "Searching todos"
    }


@app.delete("/todo/{todo_id}")
def delete_todo(todo_id: int):
    return {
        "message": "Todo deleted",
        "todo_id": todo_id
    }


@app.put("/todo/{todo_id}")
def update_todo(todo_id: int, todo: Todo):
    return {
        "message": "Todo updated",
        "todo_id": todo_id,
        "todo": todo
    }


@app.get("/external-todos")
def get_external_todos():
    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/todos",
            params={"_limit": 5},
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    from fastapi import HTTPException

@app.get("/todo-error/{todo_id}")
def todo_error(todo_id: int):
    if todo_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="Todo ID must be greater than 0"
        )

    return {
        "message": "Valid Todo ID",
        "todo_id": todo_id
    }
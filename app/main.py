# !/home/berna/.env/bin

from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.routers import models, datasets, users
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ai4teach")
app.include_router(users.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Mover a un router auth.py

@app.get("/", tags=["login"])
async def show_login(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

@app.post("/", tags=["login"])
async def process_login(request: Request, response: Response):
        form = await request.form()
        username = form.get("username")
        password = form.get("password") 
        if username == "a" and password == "a":
                response.set_cookie(key="usuario", value=username, httponly=True, secure=True)
                return templates.TemplateResponse("home.html", {"request": request, "login_failed": False, "username": username})
        else:
                return templates.TemplateResponse("login.html", {"request": request, "login_failed": True})

if __name__ == "__main__":
    uvicorn.run(
            "app.main:app",
            host    = "127.0.0.1",
            port    = 8080, 
            reload  = True,
    )
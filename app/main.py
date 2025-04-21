# !/home/berna/.env/bin

from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routers import models, datasets, users
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ai4teach")
app.include_router(users.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O frontend ej: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Mover a un router auth.py

@app.get("/", tags=["login"])
async def show_login(request: Request):
        return templates.TemplateResponse("login.html", {"request": request})

@app.get("/home", tags=["home"])
async def get_home(request: Request):
         username = request.cookies.get("usuario")
         return templates.TemplateResponse("home.html", {"request": request, "username": username})

@app.get("/modelos", tags=["modelos"])
async def get_home(request: Request):
         username = request.cookies.get("usuario")
         return templates.TemplateResponse("modelos.html", {"request": request, "username": username})

@app.get("/entrenamiento", tags=["entrenamiento"])
async def get_home(request: Request):
         username = request.cookies.get("usuario")
         return templates.TemplateResponse("entrenamiento.html", {"request": request, "username": username})

@app.get("/datasets", tags=["datasets"])
async def get_home(request: Request):
         username = request.cookies.get("usuario")
         return templates.TemplateResponse("datasets.html", {"request": request, "username": username})

@app.get("/contenidos", tags=["contenidos"])
async def get_home(request: Request):
         username = request.cookies.get("usuario")
         return templates.TemplateResponse("contenidos.html", {"request": request, "username": username})

@app.post("/", tags=["login"])
async def process_login(request: Request, response: Response):
        form = await request.form()
        username = form.get("username")
        password = form.get("password") 
        if username == "bernabe" and password == "bernabe":
                response = RedirectResponse(url="/home", status_code=303) 
                response.set_cookie(key="usuario", value=username, httponly=True, secure=True)
                return response
        
        return templates.TemplateResponse("login.html", {"request": request, "login_failed": True})

if __name__ == "__main__":
    uvicorn.run(
            "app.main:app",
            host    = "127.0.0.1",
            port    = 8080, 
            reload  = True,
    )
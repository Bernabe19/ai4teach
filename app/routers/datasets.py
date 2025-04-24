from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
import json

router = APIRouter()

@router.get("/temp")
async def read_datasets():
    return None

@router.get("/get-datasets")
async def get_datasets():
    try:
        file_path = Path("db/datasets.json")  # Ajusta esta ruta según tu estructura
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
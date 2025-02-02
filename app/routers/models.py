from fastapi import APIRouter

router = APIRouter()

@router.get("/temp")
async def read_models():
    return None
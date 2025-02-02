from fastapi import APIRouter

router = APIRouter()

@router.get("/temp")
async def read_datasets():
    return None
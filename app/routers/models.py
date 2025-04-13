from fastapi import APIRouter, Form, UploadFile, File, Request
from typing import Optional
import tensorflow as tf

router = APIRouter()

@router.get("/temp")
async def read_models():
    return None

@router.post("/model-params")
async def read_model_params(
    model: str = Form(...),
    dataSource: str = Form(...),
    dataset: Optional[str] = Form(None),
    format: Optional[str] = Form(None),
    batchSize: int = Form(...),
    epochs: int = Form(...),
    optimizer: str = Form(...),
    learningRate: float = Form(...),
    validationSplit: int = Form(...),
    earlyStopping: str = Form(...),
    enableAug: bool = Form(...),
    augmentation: Optional[str] = Form(None),  # JSON como string
    file: Optional[UploadFile] = File(None)
):
    
    if file:
        print("Archivo recibido:", file.filename)
    if augmentation:
        import json
        aug_data = json.loads(augmentation)
        print("Aumento de datos:", aug_data)
        
    # model_name = form.get("model_name")
    # model_type = form.get("model_type")
    # # model_params = form.get("model_params")

    # if model_type == "tf":
    #     if model_name == "resnet50":
    #         model = tf.keras.applications.resnet50.ResNet50(weights=None)
    #         # model.load_weights("../data/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    #         return {"model": str(model)}
    #     elif model_name == "mobilenet":
    #         model = tf.keras.applications.mobilenet.MobileNet(weights=None)
    #         return {"model": str(model)}
    #     else:
    #         return {"error": "Model not found"}
    # else:
    #     return {"error": "Model type not supported"}
# model = tf.keras.applications.resnet50.ResNet50(weights=None)

# model.load_weights("../data/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5")

# print(model)
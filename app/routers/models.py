from fastapi import APIRouter, Form, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import tensorflow_datasets as tfds
import time
import json
from PIL import Image
from io import BytesIO
import numpy as np
import gc

router = APIRouter()

last_training_config = {}

# Variable global para el modelo
global_model = None

# Configurar límites de memoria GPU al inicio
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar crecimiento de memoria dinámico
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU memory: {e}")

# Cargar el modelo con mejor gestión de memoria
def load_model():
    global global_model
    
    # Liberar memoria del modelo anterior si existe
    if global_model is not None:
        del global_model
        K.clear_session()
        gc.collect()
    
    
    model = models.Sequential()

    # Capa de Convolución 1
    input_shape = (28, 28, 1)  # MNIST es 28x28x1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Capa de Convolución 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Capa de Convolución 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Aplanado (flatten) para pasar a las capas densas
    model.add(layers.Flatten())
    
    # Capa densa
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))  # Dropout para evitar sobreajuste
    
    # Capa de salida
    model.add(layers.Dense(10, activation='softmax'))  # 10 clases para MNIST

    # base_model = tf.keras.applications.MobileNetV2(
    #     weights="imagenet",
    #     include_top=False,
    #     input_shape=(224, 224, 3),
    #     alpha=0.75  # Versión más ligera del modelo (75% de filtros)
    # )

    # base_model.trainable = False  # Congelar capas base

    # # Añadimos la nueva cabeza de clasificación
    # x = layers.GlobalAveragePooling2D()(base_model.output)
    # x = layers.Dense(128, activation='relu')(x)  # Reducido de 256 a 128
    # x = layers.Dropout(0.5)(x)
    # output = layers.Dense(10, activation='softmax')(x)  # MNIST = 10 clases

    # model = models.Model(inputs=base_model.input, outputs=output)
    return model

# Inicializar el modelo bajo demanda, no al inicio
def get_model():
    global global_model
    if global_model is None:
        global_model = load_model()
    return global_model

# Preprocesamiento de imágenes
def preprocess_resnet(image, label):
    # image = tf.image.resize(image, (224, 224))
    # image = tf.image.grayscale_to_rgb(image)  # MNIST es 1 canal
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Función para cargar y preparar datasets con menos consumo de memoria
def prepare_datasets():
    data_dir = "../data/datasets/"
    train_split = f"train[:{100 - last_training_config['validationSplit']}%]"
    val_split = f"train[{100 - last_training_config['validationSplit']}%:]"
    
    # Cargar datasets
    train_data = tfds.load("mnist", split=train_split, data_dir=data_dir, as_supervised=True)
    val_data = tfds.load("mnist", split=val_split, data_dir=data_dir, as_supervised=True)
    test_data = tfds.load("mnist", split="test", data_dir=data_dir, as_supervised=True)
    
    # Tomamos muestras según el porcentaje especificado
    train_sample = int(len(train_data) * (last_training_config["porcentajeDataset"] / 100)) 
    val_sample = int(len(val_data) * (last_training_config["porcentajeDataset"] / 100))
    test_sample = int(len(test_data) * (last_training_config["porcentajeDataset"] / 100))
    
    train_data = train_data.take(train_sample)
    val_data = val_data.take(val_sample)
    test_data = test_data.take(test_sample)

    # Preprocesamiento
    train_data = train_data.map(preprocess_resnet)
    val_data = val_data.map(preprocess_resnet)
    test_data = test_data.map(preprocess_resnet)

    # Aumento de datos (si se habilita)
    if last_training_config["enableAug"] and last_training_config["augmentation"]:
        aug_config = json.loads(last_training_config["augmentation"])

        def augment_fn(image, label):
            if aug_config.get("flip", False):
                image = tf.image.random_flip_left_right(image)
            if aug_config.get("rotation", 0) > 0:
                angle = tf.random.uniform([], -aug_config["rotation"], aug_config["rotation"], dtype=tf.float32)
                image = tf.image.rotate(image, angle)
            return image, label

        train_data = train_data.map(augment_fn)

    # Preparación de datos con prefetch controlado
    train_data = train_data.shuffle(1000).batch(last_training_config["batchSize"]).prefetch(1)
    val_data = val_data.batch(last_training_config["batchSize"]).prefetch(1)
    test_data = test_data.batch(last_training_config["batchSize"]).prefetch(1)
    
    return train_data, val_data, test_data

# Función de entrenamiento con streaming optimizada para memoria
def train_and_stream():
    model = get_model()
    
    # Cargamos datasets solo cuando los necesitamos
    train_data, val_data, test_data = prepare_datasets()
    
    # Configuración del optimizador
    opt = tf.keras.optimizers.get(last_training_config["optimizer"])
    opt.learning_rate = last_training_config["learningRate"]

    # Configurar early stopping si está habilitado
    callbacks = []
    if last_training_config["earlyStopping"] and last_training_config["earlyStopping"] != "none":
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

    # Compilamos el modelo
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenamiento y streaming
    try:
        for epoch in range(last_training_config["epochs"]):
            # Usar un objeto de memoria para el entrenamiento
            history = model.fit(
                train_data, 
                validation_data=val_data, 
                epochs=1, 
                verbose=0,
                callbacks=callbacks
            )
            
            # Limpiar caché entre épocas
            if hasattr(train_data, '_iterator'):
                train_data._iterator = None
            if hasattr(val_data, '_iterator'):
                val_data._iterator = None
            
            metrics = {
                "epoch": epoch + 1,
                "train_loss": float(history.history["loss"][0]),  # Convertir a float para serialización
                "val_loss": float(history.history["val_loss"][0]),
                "train_accuracy": float(history.history["accuracy"][0]),
                "val_accuracy": float(history.history["val_accuracy"][0]),
            }

            if epoch == last_training_config["epochs"] - 1:
                _, test_acc = model.evaluate(test_data, verbose=0)
                metrics["test_accuracy"] = float(test_acc)

            yield f"data: {json.dumps(metrics)}\n\n"
            
            # Pequeña pausa y liberación de memoria
            time.sleep(0.1)
            gc.collect()
    
    finally:
        # Limpieza final
        del train_data
        del val_data
        del test_data
        K.clear_session()
        gc.collect()

# Ruta para streaming de métricas
@router.get("/stream-metrics")
async def stream_metrics():
    return StreamingResponse(train_and_stream(), media_type="text/event-stream")

# Ruta para recibir parámetros de entrenamiento
@router.post("/model-params")
async def read_model_params(
    model: str = Form(...),
    dataSource: str = Form(...),
    dataset: Optional[str] = Form(None),
    format: Optional[str] = Form(None),
    porcentajeDataset: int = Form(...),
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
    global last_training_config
    last_training_config.update({
        "model": model,
        "dataSource": dataSource,
        "dataset": dataset,
        "format": format,
        "porcentajeDataset": porcentajeDataset,
        "batchSize": batchSize,
        "epochs": epochs,
        "optimizer": optimizer,
        "learningRate": learningRate,
        "validationSplit": validationSplit,
        "earlyStopping": earlyStopping,
        "enableAug": enableAug,
        "augmentation": augmentation
    })
    
    # Si cambiamos algún parámetro crítico, forzamos recarga del modelo
    if model != last_training_config.get("prev_model"):
        global global_model
        if global_model is not None:
            del global_model
            global_model = None
            K.clear_session()
            gc.collect()
        last_training_config["prev_model"] = model
    
    return JSONResponse({"message": "Parámetros recibidos correctamente"})

# Ruta para inferencia optimizada
@router.post("/run-inference")
@router.post("/run-inference")
async def read_inference(
    file: UploadFile = File(...),
):
    try:
        # Cargar modelo (o reutilizar uno existente)
        model = get_model()
        if model is None:
            return JSONResponse({"status": "error", "message": "Modelo no cargado"})

        # Leer el contenido del archivo
        contents = await file.read()
        image = Image.open(BytesIO(contents)) #.convert("L")  # Convertir a escala de grises
        image = image.resize((28, 28))
        img_array = np.array(image)

        # Normalizar y agregar dimensiones
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 28, 28, 1)

        # Medir el tiempo de inferencia
        start_time = time.time()
        predictions = model.predict(img_array)
        end_time = time.time()

        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        inference_time = end_time - start_time

        # Opcional: confianza por clase
        confidences = {f"class_{i}": float(pred) for i, pred in enumerate(predictions[0])}

        return JSONResponse({
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "inference_time": round(inference_time, 4),
            "confidences": confidences
        })
    
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# Ruta para liberar explícitamente la memoria (útil para debugging)
@router.post("/clear-memory")
async def clear_memory():
    global global_model
    if global_model is not None:
        del global_model
        global_model = None
    
    K.clear_session()
    gc.collect()
    
    return JSONResponse({"message": "Memoria liberada"})
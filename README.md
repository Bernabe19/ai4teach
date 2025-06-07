
# 📦 AI4Teach - Despliegue Local de la Aplicación

Esta guía proporciona los pasos necesarios para desplegar **AI4Teach** en un entorno local. La aplicación permite al alumnado entrenar e inferir modelos de IA con una interfaz sencilla y educativa. El backend está desarrollado con **FastAPI** y utiliza **TensorFlow con soporte CUDA** para aceleración por GPU. El frontend está construido con **HTML, CSS y JavaScript**.

---

## ✅ Requisitos previos

Antes de iniciar el despliegue, asegúrate de que el equipo donde se instalará cumple con los siguientes requisitos:

- Sistema operativo: Linux, Windows o macOS
- Python 3.8 o superior
- NVIDIA GPU con drivers compatibles
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) y cuDNN instalados
- Navegador web moderno (Chrome, Firefox, Edge)

---

## 🔧 Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/AI4Teach.git
cd AI4Teach
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Asegúrate de tener instalados `tensorflow`, `tensorflow-gpu` y `fastapi`.

Si usas GPU:

```bash
pip install tensorflow-gpu
```

### 4. Verificar instalación de CUDA

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Debe devolver algo como:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🚀 Ejecutar la aplicación

### 1. Lanzar el backend (FastAPI)

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Este comando inicia el backend en la IP local, escuchando por el puerto `8000`.

### 2. Acceder al frontend

Abre directamente el archivo `login.html` ubicado en la carpeta `frontend` desde el navegador de los equipos cliente. No necesitas servir el frontend si accedes desde la misma red.


---

## 📁 Estructura del proyecto

```
AI4Teach/
│
├── app                   # Directorio principal 
│   ├── routers/
│   ├── static/
│   ├── templates/
│   └── main.py
│
├── data/                 # Modelos-Datos-Usuarios
│   ├── datasets/           
│   ├── models/           
│   └── usuarios/
│
├── start/                # Ejemplos para pruebas
│   ├── cnn_model.h5        
│   ├── download_datasets.py
│   ├── download_models.py
│   ├── test_image.png
│   └── test_model.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Notas para docentes

- La aplicación puede usarse desde cualquier ordenador conectado a la misma red local mientras el backend esté en ejecución.
- Los modelos entrenados pueden exportarse y reutilizarse en otros entornos (TensorFlow.js, TFLite, etc.).
- Una vez desplegada en el dispositivo del docente, la aplicación puede ser accedida por el alumnado desde otros dispositivos conectados a la misma red local (por ejemplo, la red WiFi del centro).
- No se requiere conexión a Internet para el uso habitual de la aplicación, ya que todo el procesamiento (entrenamiento, inferencia, visualización) se realiza de forma local.



---

## 📌 Próximamente

- Despliegue en la nube con Azure AI (aprovechando convenios con Conselleria)
- Sistema de gestión docente y de progreso del alumnado
- Ranking colaborativo de modelos entre clases o centros

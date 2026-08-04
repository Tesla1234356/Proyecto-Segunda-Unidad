<div align="center">

# 🤖 SENTINEL AI
### Sistema de Vigilancia Autónoma con Visión Artificial

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![DeepFace](https://img.shields.io/badge/DeepFace-FaceNet-00BCD4?style=for-the-badge)](https://github.com/serengil/deepface)
[![License](https://img.shields.io/badge/Licencia-Académica-green?style=for-the-badge)](LICENSE)

> **Proyecto Final - Segunda Unidad | Curso de Visión Artificial**  
> Propuesta B: Robot Móvil Autónomo de Seguridad/Servicio

---

</div>

## 📌 ¿Qué es Sentinel AI?

**Sentinel AI** es un sistema de vigilancia inteligente que simula el cerebro de un robot de seguridad autónomo. Integra **cuatro técnicas fundamentales de Visión Artificial** funcionando en tiempo real sobre una sola máquina, capaz de:

- 👁️ **Ver** el entorno e identificar objetos y personas
- 🧠 **Recordar** quién es quién con biometría facial
- 🗺️ **Saber dónde está** sin necesidad de GPS
- 🚨 **Tomar decisiones** autónomas de alerta o acceso

---

## 🎯 Funcionalidades Principales

### 1. 👁️ Detección de Objetos — *Los Ojos del Robot*
> Motor: **YOLOv8 Nano** | Dataset: **COCO (80 categorías)**

El robot detecta y clasifica objetos en tiempo real usando redes neuronales convolucionales. Identifica obstáculos como sillas y mesas (para no chocar), y objetos de valor como laptops o celulares (para vigilarlos).

```
Persona detectada → ID asignado → Tracking activado
Objeto de valor   → Alerta visual en el HUD
```

---

### 2. 🔁 Seguimiento / Tracking — *La Memoria Visual*
> Algoritmo: **ByteTrack** vía `Ultralytics .track(persist=True)`

Una vez detectada una persona, el sistema le asigna un **ID único persistente**. Si el sujeto desaparece detrás de un obstáculo y reaparece, el sistema lo reconoce como el **mismo individuo** y no como uno nuevo.

```
Intruso detectado [ID: 1] → Se oculta → Reaparece → Sigue siendo [ID: 1]
```

---

### 3. 🗺️ Odometría Visual — *El GPS de Interiores*
> Algoritmo: **ORB + Matriz Esencial + RANSAC + Filtro EMA**

Como el GPS no funciona en interiores, el robot usa su cámara para estimar su propio movimiento. Analiza el desplazamiento de píxeles entre fotogramas para trazar su ruta en un **mini-mapa radar 2D** en el HUD.

| Componente | Función |
|---|---|
| **ORB** (500 puntos) | Extrae puntos clave del entorno |
| **RANSAC** | Filtra correspondencias erróneas |
| **Matriz Esencial** | Calcula vectores de traslación |
| **Filtro EMA** (α=0.10) | Estabiliza el ruido de la cámara |

---

### 4. 👤 Reconocimiento Facial — *El Guardia Biométrico*
> Motor: **FaceNet (Google)** vía `DeepFace` | Embeddings: **128 dimensiones**

El sistema escanea los rostros detectados y los compara contra la base de datos local (`data/rostros/`). Calcula la Distancia Euclidiana/Coseno del vector de características para determinar si la persona es:

- ✅ **Personal VIP / Autorizado** → Acceso permitido, etiqueta verde
- ⚠️ **Desconocido** → Registrado como sospechoso, alerta en el HUD

---

## 🏗️ Arquitectura del Sistema

```
sentinel-ai/
│
├── main.py                  ← Director de orquesta (bucle principal)
│
├── modules/
│   ├── core.py              ← Motor central (SentinelCore) — coordina todo
│   ├── decision.py          ← Cerebro autónomo de decisiones (MotorDecision)
│   ├── odometria.py         ← GPS de interiores (OdometriaORB)
│   ├── reconocimiento.py    ← Biometría facial (ReconocimientoFacial)
│   ├── hud.py               ← Interfaz HUD en pantalla (crear_hud_base)
│   ├── launcher.py          ← Pantalla de carga y menú de inicio
│   ├── evidencias.py        ← Gestor de capturas y evidencias
│   └── audio.py             ← Gestor de alertas sonoras
│
├── models/
│   └── yolov8n.pt           ← Modelo YOLOv8 Nano (no incluido en repo)
│
├── data/
│   ├── rostros/             ← Base de datos biométrica local
│   └── sonidos/             ← Archivos de audio para alertas
│
├── tests/                   ← Scripts de prueba por módulo
├── evidencias/              ← Capturas generadas en tiempo de ejecución
├── requirements.txt
└── mi_tracker.yaml          ← Configuración de tracker personalizado
```

---

## ⚡ Optimizaciones de Rendimiento

Correr 3 redes neuronales simultáneamente en una PC convencional requiere estrategias específicas. Se implementaron **3 técnicas de nivel de producción**:

| Técnica | Descripción | Impacto |
|---|---|---|
| **Redimensionado Dinámico** | Videos 1080p/4K reducidos a máx. 600px manteniendo aspecto | ↓ 80-90% carga CPU |
| **Delegación CUDA** | YOLOv8 procesado en GPU (GTX 1650) con `device=0` | CPU libre para otros módulos |
| **Frame Skipping** | FaceNet analiza solo 1 de cada 10 fotogramas, resultado interpolado visualmente | Sin cuello de botella biométrico |

---

## 🚀 Instalación y Uso

### Prerrequisitos
- Python 3.10+
- GPU NVIDIA con CUDA (recomendado) o CPU potente
- Cámara web, cámara IP o archivo de video

### 1. Clonar el repositorio
```bash
git clone https://github.com/Tesla1234356/Proyecto-Segunda-Unidad.git
cd Proyecto-Segunda-Unidad
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el modelo YOLOv8
```bash
# Se descarga automáticamente al primer uso, o manualmente:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Mover el archivo descargado a la carpeta models/
```

### 4. Ejecutar el sistema
```bash
python main.py
```

Al iniciar, aparecerá una **pantalla de carga** con progreso, seguida de un **menú de selección** donde puedes elegir:
- 📷 Cámara web local
- 📱 Cámara IP del celular (vía URL HTTP)
- 🎬 Archivo de video MP4

---

## 🎮 Controles en Tiempo de Ejecución

| Tecla | Acción |
|---|---|
| `Q` | Apagar el sistema |

---

## 🔧 Fuentes de Entrada Soportadas

```python
# Cámara web (puerto 0 o 1)
cv2.VideoCapture(0)

# Cámara IP del celular (IP Webcam App)
cv2.VideoCapture("http://192.168.x.x:8080/video")
# Nota: se aplica espejo automático para cámaras IP

# Archivo de video
cv2.VideoCapture("ruta/al/video.mp4")
```

---

## 📦 Dependencias

```
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
ultralytics==8.0.200      # YOLOv8 + ByteTrack
deepface==0.0.79          # FaceNet (reconocimiento facial)
tf-keras==2.15.0          # Backend para DeepFace
filterpy==1.4.5           # Algoritmos de tracking
lapx==0.5.5               # Asignaciones matemáticas (tracking)
numpy==1.26.4
pandas==2.1.4
```

---

## 📋 Tests por Módulo

En la carpeta `tests/` se encuentran scripts independientes para validar cada etapa del sistema:

| Script | Módulo que prueba |
|---|---|
| `etapa1_deteccion_tracking.py` | YOLOv8 + ByteTrack |
| `etapa2_odometria.py` | ORB + Odometría visual |
| `etapa3_reconocimiento.py` | FaceNet + Reconocimiento facial |

---

## 👨‍💻 Tecnologías Utilizadas

<div align="center">

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.10+ | Lenguaje base |
| OpenCV | 4.8 | Procesamiento de imagen y video |
| Ultralytics YOLOv8 | 8.0.200 | Detección de objetos + Tracking |
| DeepFace / FaceNet | 0.0.79 | Reconocimiento facial biométrico |
| NumPy | 1.26.4 | Operaciones matriciales |
| Tkinter | stdlib | Interfaz de launcher/menú |

</div>

---

## 📄 Documentación Adicional

- [`README_ROBOT_CASOUSO.md`](README_ROBOT_CASOUSO.md) — Explicación funcional desde la perspectiva del robot
- [`README_TECNICO.md`](README_TECNICO.md) — Documentación técnica detallada de cada algoritmo

---

<div align="center">

**Desarrollado para la presentación final del Curso de Visión Artificial** 🎓

</div>

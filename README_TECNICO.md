# Sentinel AI - Documentación Técnica 🛠️🧠

Este documento detalla la implementación técnica del proyecto para la **Segunda Unidad del curso de Visión Artificial** (Propuesta B).

---

## 🎯 Cumplimiento Técnico de la Rúbrica

### 1. Detección de Objetos (YOLOv8)
*   **Requisito:** "Identifica obstáculos (sillas, mesas) y objetos de interés (teléfonos, portátiles)."
*   **Solución:** Utilizamos el modelo **YOLOv8 Nano** entrenado con el dataset COCO (80 categorías). El sistema detecta en tiempo real y clasifica objetos a través de redes neuronales convolucionales (CNN), permitiendo al sistema comprender su entorno inmediato.

### 2. Seguimiento / Tracking (ByteTrack)
*   **Requisito:** "Garantiza que si un objeto desaparece y reaparece, el robot mantenga la misma identificación."
*   **Solución:** Incorporamos el algoritmo de seguimiento (Tracking) usando el parámetro `.track(persist=True)` de Ultralytics. Se le asigna un **ID único** a cada bounding box, manteniendo un historial matemático de las trayectorias para predecir colisiones o re-identificar objetos.

### 3. Odometría Visual (ORB + Matriz Esencial)
*   **Requisito:** "Permite al robot navegar y mapear la habitación estimando su propio movimiento."
*   **Solución:** Implementamos un HUD 2D. Se usa el extractor **ORB** (Oriented FAST and Rotated BRIEF) limitando a 500 puntos para optimizar la velocidad. Mediante *RANSAC* y la *Matriz Esencial* (`cv2.findEssentialMat`), se calculan los vectores de traslación. Se aplicó un **Filtro de Suavizado EMA** (Exponential Moving Average) con alpha=0.10 para estabilizar el ruido de la cámara.

### 4. Reconocimiento Facial (FaceNet)
*   **Requisito:** "Detecta rostros para diferenciar al personal autorizado de los desconocidos."
*   **Solución:** Integramos `DeepFace` utilizando el modelo **FaceNet** (Google). El modelo extrae Embeddings de 128 dimensiones del rostro y calcula la Distancia Euclidiana/Coseno contra el dataset local (`data/rostros/`) para la validación biométrica.

---

## ⚙️ Arquitectura de Optimización (Factibilidad)

Para ejecutar las 3 Redes Neuronales pesadas simultáneamente en una sola PC (CPU/GPU híbrido) sin perder FPS, se aplicaron 3 soluciones de nivel de producción:

1.  **Redimensionado Dinámico (Anti-Lag):** Los videos de entrada (ej. 1080p) se reducen proporcionalmente a un máximo de 600 píxeles por lado, reduciendo la carga de cálculo de la CPU en más de un 80%.
2.  **Delegación de Hardware (CUDA):** Se forzó a YOLOv8 a procesar los tensores en la GPU (GTX 1650) mediante el parámetro `device=0`.
3.  **Frame Skipping (Salto de Fotogramas):** Dado que FaceNet corre sobre la CPU y genera cuellos de botella, se programó para analizar biometría solo **1 de cada 10 fotogramas**, interpolando los resultados visualmente para mantener una interfaz fluida sin parpadeos.

# Sentinel AI - Sistema de Visión Artificial Híbrido 🤖👁️

Este proyecto es la solución desarrollada para la **Segunda Unidad del curso de Visión Artificial**. Implementa la **Propuesta B** (Robot móvil autónomo de seguridad/servicio) exigida en la rúbrica, integrando cuatro técnicas fundamentales de la IA en un solo sistema funcional.

---

## 🎯 Objetivos del Proyecto y Cumplimiento de la Rúbrica

El sistema "Sentinel AI" cumple estrictamente con las 4 funcionalidades requeridas por el profesor:

### 1. Detección de Objetos (Los Ojos del Robot)
*   **¿Para qué le sirve al robot?** Cuando el robot patrulla por una oficina, necesita "ver". Usando **YOLOv8**, el robot identifica obstáculos en el piso (como sillas o mesas) para no chocar contra ellos. Además, detecta objetos de valor (como laptops o celulares) para vigilar que nadie se los robe.

### 2. Seguimiento / Tracking (La Memoria Visual del Robot)
*   **¿Para qué le sirve al robot?** Si el robot detecta a un intruso corriendo, necesita no perderlo de vista. Con **ByteTrack**, el robot le asigna un número de ID único al intruso. Si el ladrón se esconde detrás de una pared o una columna y vuelve a salir, el robot ya sabe que es el *mismo* intruso (ID: 1) y no una persona nueva.

### 3. Odometría Visual (El GPS de Interiores del Robot)
*   **¿Para qué le sirve al robot?** Como el GPS normal no funciona dentro de los edificios, el robot usa su cámara como brújula y mapa. Con el algoritmo **ORB**, el robot analiza cómo se mueven los píxeles del suelo para trazar una línea roja en su radar. Así, el robot sabe exactamente dónde empezó su patrullaje, qué pasillos ya recorrió y dónde está parado en ese momento.

### 4. Reconocimiento Facial (El Guardia de Seguridad del Robot)
*   **¿Para qué le sirve al robot?** Cuando el robot se cruza con una persona, necesita saber si debe saludarla o sonar una alarma. Usando **FaceNet**, el robot escanea la cara y busca en su base de datos. Si eres personal autorizado (VIP), el robot te deja pasar tranquilo. Si eres un desconocido, el robot sabe que debe registrarte como sospechoso.

---

## ⚙️ Arquitectura del Sistema (main.py)

El archivo `main.py` actúa como el **Director de Orquesta**, sincronizando los 3 módulos:

1.  `modules/odometria.py`
2.  `modules/reconocimiento.py`
3.  *YOLOv8 integrado directamente en el main.*

### 🛠️ Optimizaciones Técnicas (Factibilidad de Producción)
Correr 3 redes neuronales pesadas simultáneamente destruiría los recursos de una PC normal. Para hacerlo factible en hardware no comercial, aplicamos 3 técnicas de optimización:

1.  **Redimensionado Dinámico (Anti-Lag):** Los videos 1080p o 4K se reducen automáticamente manteniendo su relación de aspecto (*Aspect Ratio*) a una resolución máxima de 600px. Esto reduce la carga del CPU en un 90%.
2.  **Aceleración Híbrida (CPU + GPU):** Obligamos a YOLOv8 a procesarse en la Tarjeta Gráfica (GTX 1650 con CUDA usando `device=0`), liberando al procesador.
3.  **Frame Skipping Biométrica:** Como FaceNet procesa en CPU y es un modelo pesado, se programó para ejecutarse solo **1 de cada 10 fotogramas** (`if frame_count % 10 == 0:`). Sin embargo, el cuadro VIP se mantiene persistente en pantalla para evitar el parpadeo visual.

---
**Desarrollado para la presentación final de Visión Artificial - 2026**

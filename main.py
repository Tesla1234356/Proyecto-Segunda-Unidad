import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from modules.odometria import OdometriaORB
from modules.reconocimiento import ReconocimientoFacial
from modules.hud import crear_hud_base
from modules.launcher import mostrar_launcher
from modules.evidencias import GestorEvidencias
from modules.audio import GestorAudio
from modules.decision import MotorDecision
from modules.core import SentinelCore

# Diccionario COCO traducido al español para YOLOv8
NOMBRES_ESPANOL = {
    0: 'persona', 1: 'bicicleta', 2: 'coche', 3: 'motocicleta', 4: 'avion', 5: 'autobus', 6: 'tren', 7: 'camion', 8: 'barco', 9: 'semaforo',
    10: 'boca de incendios', 11: 'señal de stop', 12: 'parquimetro', 13: 'banco', 14: 'pajaro', 15: 'gato', 16: 'perro', 17: 'caballo', 18: 'oveja', 19: 'vaca',
    20: 'elefante', 21: 'oso', 22: 'cebra', 23: 'jirafa', 24: 'mochila', 25: 'paraguas', 26: 'bolso', 27: 'corbata', 28: 'maleta', 29: 'frisbee',
    30: 'esquis', 31: 'snowboard', 32: 'pelota', 33: 'cometa', 34: 'bate', 35: 'guante', 36: 'monopatin', 37: 'tabla de surf', 38: 'raqueta', 39: 'botella',
    40: 'copa', 41: 'taza', 42: 'tenedor', 43: 'cuchillo', 44: 'cuchara', 45: 'bol', 46: 'platano', 47: 'manzana', 48: 'sandwich', 49: 'naranja',
    50: 'brocoli', 51: 'zanahoria', 52: 'perrito caliente', 53: 'pizza', 54: 'donut', 55: 'pastel', 56: 'silla', 57: 'sofa', 58: 'planta', 59: 'cama',
    60: 'mesa', 61: 'inodoro', 62: 'monitor', 63: 'portatil', 64: 'raton', 65: 'mando', 66: 'teclado', 67: 'celular', 68: 'microondas', 69: 'horno',
    70: 'tostadora', 71: 'fregadero', 72: 'nevera', 73: 'libro', 74: 'reloj', 75: 'florero', 76: 'tijeras', 77: 'peluche', 78: 'secadora', 79: 'cepillo'
}

def iniciar_sentinel_ai():
    print("=======================================")
    print(" INICIANDO SISTEMA SENTINEL AI - V2.0  ")
    print("=======================================")
    
    # Contenedores para extraer los modelos del hilo de Tkinter
    modelos = {}

    def cargador_callback(lbl_status, progress, root):
        lbl_status.config(text="[1/3] Cargando Motor de Detección (YOLOv8)...")
        progress['value'] = 25
        root.update()
        modelos['yolo'] = YOLO("models/yolov8n.pt")
        
        lbl_status.config(text="[2/3] Cargando Motor de Odometría (ORB)...")
        progress['value'] = 50
        root.update()
        modelos['odometria'] = OdometriaORB()
        
        lbl_status.config(text="[3/3] Cargando Motor Biométrico (FaceNet)...")
        progress['value'] = 75
        root.update()
        modelos['reconocedor'] = ReconocimientoFacial()
        
        lbl_status.config(text="Iniciando Sistemas de Audio y HUD...")
        progress['value'] = 100
        root.update()
        modelos['audio'] = GestorAudio()
        modelos['evidencias'] = GestorEvidencias()
        modelos['decision'] = MotorDecision()  # Cerebro de decisión autónoma
        modelos['hud_base'] = crear_hud_base()
        time.sleep(0.3) # Pequeña pausa para que el usuario vea el 100%
    
    # --- 1. MOSTRAR PANTALLA DE CARGA Y LUEGO MENÚ ---
    ruta_video = mostrar_launcher(cargador_callback)
    
    if ruta_video is None:
        print("[!] Sistema abortado por el usuario.")
        return
        
    print(f"[+] Fuente de entrada seleccionada: {ruta_video}")
    # Inicializar el Cerebro de la Inteligencia Artificial (Core Engine)
    core = SentinelCore(modelos, NOMBRES_ESPANOL)
        
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened() and isinstance(ruta_video, int):
        puerto_alterno = 0 if ruta_video == 1 else 1
        print(f"[!] Error en cámara {ruta_video}. Buscando cámara alternativa en puerto {puerto_alterno}...")
        cap = cv2.VideoCapture(puerto_alterno)
    
    print("\n>> SISTEMA ONLINE. Presiona 'q' para apagar el sistema.")
    cv2.namedWindow("Sentinel AI - Dashboard Operativo", cv2.WINDOW_NORMAL)
    
    # Determinar si es una transmisión desde el celular (IP Cam) para aplicar el Espejo
    es_camara_celular = isinstance(ruta_video, str) and ruta_video.startswith("http")

    while cap.isOpened():
        start_time = time.time()
        success, frame_raw = cap.read()
        if not success:
            print("Fin del video o cámara desconectada.")
            break
            
        if es_camara_celular:
            frame_raw = cv2.flip(frame_raw, 1) # Voltear horizontalmente (Efecto Espejo) SOLO para Celular
            
            
        # --- EL CEREBRO DE LA IA PROCESA EL FOTOGRAMA ---
        hud_final = core.ejecutar_ciclo(frame_raw, start_time)
        
        cv2.imshow("Sentinel AI - Dashboard Operativo", hud_final)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("SISTEMA APAGADO.")

if __name__ == '__main__':
    iniciar_sentinel_ai()
import cv2
import time
import os
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from ultralytics import YOLO
from modules.odometria import OdometriaORB
from modules.reconocimiento import ReconocimientoFacial
from modules.hud import crear_hud_base, actualizar_hud

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
    
    if not os.path.exists("evidencias"):
        os.makedirs("evidencias")
        
    print("[1/3] Cargando Motor de Detección (YOLOv8)...")
    yolo_model = YOLO("models/yolov8n.pt")
    
    print("[2/3] Cargando Motor de Odometría (ORB)...")
    odometria = OdometriaORB()
    
    print("[3/3] Cargando Motor Biométrico (FaceNet)...")
    reconocedor = ReconocimientoFacial()
    
    # Pre-crear la interfaz gráfica para no consumir CPU en el bucle
    hud_base = crear_hud_base()
    
    # ¡MAGIA GOZU! Selector Interactivo de Video (Interfaz Nativa)
    print("\n[!] Abre la ventana emergente para seleccionar tu video de prueba...")
    root = tk.Tk()
    root.withdraw() # Ocultamos la ventana principal aburrida
    ruta_video = filedialog.askopenfilename(
        title="Sentinel AI - Cargar Archivo de Seguridad",
        filetypes=[("Archivos de Video", "*.mp4 *.avi *.mkv *.mov"), ("Todos los archivos", "*.*")]
    )
    
    if not ruta_video:
        print("[X] Operación cancelada. Usando cámara web por defecto (0)...")
        ruta_video = 0
    else:
        print(f"[+] Archivo cargado con éxito: {os.path.basename(ruta_video)}")
        
    cap = cv2.VideoCapture(ruta_video)
    
    frame_count = 0
    ultimo_rostro = None
    alertas = ["[!] Sistema Iniciado"]
    registros_id = set()
    hacker_log = [] # NUEVO: Lista persistente para el scroll del terminal hacker
    
    print("\n>> SISTEMA ONLINE. Presiona 'q' para apagar el sistema.")
    
    # ¡MAGIA GOZU! Habilitar que la ventana se pueda maximizar y estirar sin romperse
    cv2.namedWindow("Sentinel AI - Dashboard Operativo", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        start_time = time.time()
        success, frame_raw = cap.read()
        frame_count += 1
        if not success:
            print("Fin del video o cámara desconectada.")
            break
            
        # --- REDIMENSIONADO ANTI-LAG (Conserva Proporción Original) ---
        h, w = frame_raw.shape[:2]
        escala = 600 / max(h, w)
        nuevo_w, nuevo_h = int(w * escala), int(h * escala)
        frame = cv2.resize(frame_raw, (nuevo_w, nuevo_h))
            
        frame_count += 1
        
        # --- A. ODOMETRÍA ---
        _, radar_map = odometria.procesar_frame(frame)
        
        # --- B. YOLO + BoT-SORT (Tracking Avanzado con Re-ID) ---
        # conf=0.35 para que YOLO procese todo lo que crea que es un objeto (luego lo filtramos)
        resultados_yolo = yolo_model.track(frame, persist=True, tracker="botsort.yaml", conf=0.35, verbose=False, device=0)
        
        frame_anotado = frame.copy()
        objetos_en_pantalla = 0
        
        # --- DIBUJADO MANUAL Y SISTEMA DE EVIDENCIAS (CON UMBRALES) ---
        if resultados_yolo[0].boxes.id is not None:
            boxes = resultados_yolo[0].boxes.xyxy.cpu().numpy()
            classes = resultados_yolo[0].boxes.cls.cpu().numpy()
            ids = resultados_yolo[0].boxes.id.cpu().numpy()
            confs = resultados_yolo[0].boxes.conf.cpu().numpy()
            
            for box, cls, obj_id, conf in zip(boxes, classes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cls_int = int(cls)
                obj_id_int = int(obj_id)
                
                # --- LÓGICA DE UMBRALES Y CLASIFICACIÓN (MAGIA GOZU) ---
                if conf >= 0.65:
                    nombre_obj = NOMBRES_ESPANOL.get(cls_int, "DESCONOCIDO").upper()
                    color_caja = (0, 200, 255)
                    objetos_en_pantalla += 1
                    
                    # Generar texto hacker persistente (Se actualiza 6 veces por segundo para dar efecto SCROLL)
                    if frame_count % 5 == 0:
                        hex_code = hex(obj_id_int * 7919)[2:].upper().zfill(4)
                        hacker_log.append(f"> [{obj_id_int:02d}] {nombre_obj[:9]} {int(conf*100)}% 0x{hex_code}")
                        
                        # Si la lista se llena, eliminar el más viejo (Efecto de ruede hacia arriba)
                        if len(hacker_log) > 22:
                            hacker_log.pop(0)
                    
                    h_f, w_f = frame.shape[:2]
                    x_center = (x1 + x2) / 2
                    offset_x = (x_center / w_f) - 0.5 
                    
                    odometria.registrar_objeto_en_radar(f"{nombre_obj} ({obj_id_int})", offset_x)
                    
                    if cls_int in [0, 24, 67]: 
                        if obj_id_int not in registros_id:
                            fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            cv2.imwrite(f"evidencias/EVID_{nombre_obj}_ID{obj_id_int}_{fecha_hora}.jpg", frame_raw)
                            alertas.append(f"> LOG: {nombre_obj} detectado (ID:{obj_id_int})")
                            
                            csv_path = "evidencias/historial_objetos.csv"
                            archivo_existe = os.path.exists(csv_path)
                            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                if not archivo_existe:
                                    writer.writerow(['Fecha_Hora', 'ID_Tracking', 'Objeto', 'Confianza_Deteccion'])
                                writer.writerow([fecha_hora, obj_id_int, nombre_obj, f"{int(conf*100)}%"])
                                
                            registros_id.add(obj_id_int)
                            
                elif conf >= 0.40:
                    nombre_obj = "OBJ. DESCONOCIDO"
                    color_caja = (100, 100, 100)
                    objetos_en_pantalla += 1
                else:
                    continue 
                
                cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color_caja, 2)
                cv2.putText(frame_anotado, f"[{obj_id_int}] {nombre_obj} {int(conf*100)}%", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja, 2)

        
        # --- C. RECONOCIMIENTO FACIAL (MULTICLASE) ---
        if frame_count % 10 == 0:
            nuevo_rostro = reconocedor.reconocer_frame(frame)
            if nuevo_rostro is not None:
                ultimo_rostro = nuevo_rostro
                nombre_vip, clase_vip = nuevo_rostro[0], nuevo_rostro[1]
                alerta_vip = f"> {clase_vip}: {nombre_vip}"
                if alerta_vip not in alertas:
                    alertas.append(alerta_vip)
                
        if ultimo_rostro is not None:
            nombre, clase, x, y, w, h = ultimo_rostro
            
            # Asignar color y formato según la clase
            if clase == "SOSPECHOSO":
                color_rostro = (0, 0, 255) # ROJO ALARMA
                texto_rostro = f"[!] PELIGRO: {nombre}"
            elif clase == "CLIENTE":
                color_rostro = (255, 200, 0) # CYAN
                texto_rostro = f"[?] CLIENTE: {nombre}"
            else: # AUTORIZADO
                color_rostro = (0, 255, 0) # VERDE VIP
                texto_rostro = f"[+] STAFF: {nombre}"
                
            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), color_rostro, 3)
            cv2.putText(frame_anotado, texto_rostro, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_rostro, 2)
        
        # Calculo de FPS reales
        fps = 1.0 / (time.time() - start_time + 0.0001)
        
        # --- ENSAMBLAR Y MOSTRAR HUD IRON MAN ---
        hud_final = actualizar_hud(hud_base.copy(), frame_anotado, radar_map, fps, objetos_en_pantalla, alertas, hacker_log)
        
        cv2.imshow("Sentinel AI - Dashboard Operativo", hud_final)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("SISTEMA APAGADO.")

if __name__ == '__main__':
    iniciar_sentinel_ai()

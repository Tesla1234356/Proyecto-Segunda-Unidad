import cv2
import time
from modules.hud import actualizar_hud

class SentinelCore:
    """Motor Principal de la Inteligencia Artificial (Core Engine).
    Orquesta la comunicación entre YOLO, FaceNet, Odometría, Audio y el HUD."""
    
    def __init__(self, modelos, nombres_espanol):
        self.yolo_model = modelos['yolo']
        self.odometria = modelos['odometria']
        self.reconocedor = modelos['reconocedor']
        self.audio = modelos['audio']
        self.evidencias = modelos['evidencias']
        self.hud_base = modelos['hud_base']
        self.decision_motor = modelos['decision']  # Cerebro de decisión autónoma
        self.nombres_espanol = nombres_espanol
        
        # Estado Interno (Memoria a corto plazo)
        self.frame_count = 0
        self.alertas = ["[!] Sistema Iniciado"]
        self.hacker_log = []
        self.memoria_identidades = {}
        self.registro_biometrico_fijo = {}
        self.ultima_accion_loggeada = None

    def ejecutar_ciclo(self, frame_raw, start_time):
        """Ejecuta un ciclo completo de pensamiento del robot por cada fotograma."""
        self.frame_count += 1
        h, w = frame_raw.shape[:2]
        
        # 1. Redimensionado Inteligente (HUD: 800x600 px)
        escala = min(600 / h, 800 / w)
        nuevo_w, nuevo_h = int(w * escala), int(h * escala)
        frame = cv2.resize(frame_raw, (nuevo_w, nuevo_h))
        
        # 2. Odometría Espacial (Radar)
        _, radar_map = self.odometria.procesar_frame(frame)
        
        # 3. Detección de Objetos (YOLOv8 + ByteTrack)
        resultados_yolo = self.yolo_model.track(frame, persist=True, tracker="mi_tracker.yaml", conf=0.35, verbose=False, device=0)
        
        # 4. Fusión de Sensores (Cuerpos + Rostros)
        self._fusion_biometrica(frame, resultados_yolo)
        
        # 5. Dibujado y Clasificación de Amenazas
        frame_anotado, sospechoso_en_pantalla, objetos_en_pantalla = self._clasificar_amenazas(frame, frame_raw, resultados_yolo)
        
        # 6. DECISIÓN AUTÓNOMA (El robot decide qué hacer con lo que vio)
        h_f, w_f = frame.shape[:2]
        telemetria = self.odometria.obtener_telemetria_flujo()  # navegación SIN YOLO
        accion, color_accion, detalle_accion = self.decision_motor.decidir_accion(
            resultados_yolo, sospechoso_en_pantalla, w_f, h_f, telemetria_flujo=telemetria
        )
        if accion != self.ultima_accion_loggeada:
            self.alertas.append(f"> DECISION: {accion}")
            self.ultima_accion_loggeada = accion
        
        # 7. Control de Audio
        self.audio.procesar_alarma(sospechoso_en_pantalla)
        
        # 8. Generación del HUD Final
        fps = 1.0 / (time.time() - start_time + 0.0001)
        hud_final = actualizar_hud(
            self.hud_base.copy(), frame_anotado, radar_map, fps, objetos_en_pantalla,
            self.alertas, self.hacker_log, accion, color_accion, detalle_accion
        )
        
        return hud_final
        
    def _fusion_biometrica(self, frame, resultados_yolo):
        """Si detecta humanos, enciende FaceNet para ponerles nombre y clase."""
        if self.frame_count % 10 == 0:
            hay_personas = False
            if resultados_yolo[0].boxes.id is not None:
                if 0 in resultados_yolo[0].boxes.cls.cpu().numpy():
                    hay_personas = True
                    
            if hay_personas:
                nuevo_rostro = self.reconocedor.reconocer_frame(frame)
                if nuevo_rostro is not None:
                    nombre_vip, clase_vip, x_f, y_f, w_f, h_f = nuevo_rostro
                    centro_cara_x = x_f + w_f/2
                    centro_cara_y = y_f + h_f/2
            
                    if resultados_yolo[0].boxes.id is not None:
                        boxes = resultados_yolo[0].boxes.xyxy.cpu().numpy()
                        ids = resultados_yolo[0].boxes.id.cpu().numpy()
                        classes = resultados_yolo[0].boxes.cls.cpu().numpy()
                        
                        for box, obj_id, cls in zip(boxes, ids, classes):
                            if int(cls) == 0:
                                x1, y1, x2, y2 = box
                                if x1 <= centro_cara_x <= x2 and y1 <= centro_cara_y <= y2:
                                    self.memoria_identidades[int(obj_id)] = (nombre_vip, clase_vip)
                                    
                                    # La trampa del ID permanente
                                    if nombre_vip not in self.registro_biometrico_fijo:
                                        self.registro_biometrico_fijo[nombre_vip] = int(obj_id)
                                        
                                    id_visual_permanente = self.registro_biometrico_fijo[nombre_vip]
                                    alerta_vip = f"> BINDING: {clase_vip} {nombre_vip} -> ID:{id_visual_permanente}"
                                    if alerta_vip not in self.alertas:
                                        self.alertas.append(alerta_vip)
                                    break

    def _clasificar_amenazas(self, frame, frame_raw, resultados_yolo):
        """Pinta las cajas de colores, detecta intrusos y manda a guardar evidencias."""
        frame_anotado = frame.copy()
        objetos_en_pantalla = 0
        sospechoso_en_pantalla = False
        
        if resultados_yolo[0].boxes.id is not None:
            boxes = resultados_yolo[0].boxes.xyxy.cpu().numpy()
            classes = resultados_yolo[0].boxes.cls.cpu().numpy()
            ids = resultados_yolo[0].boxes.id.cpu().numpy()
            confs = resultados_yolo[0].boxes.conf.cpu().numpy()
            
            for box, cls, obj_id, conf in zip(boxes, classes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cls_int = int(cls)
                obj_id_int = int(obj_id)
                
                if conf >= 0.65:
                    nombre_obj = self.nombres_espanol.get(cls_int, "DESCONOCIDO").upper()
                    color_caja = (0, 200, 255)
                    
                    if cls_int == 0 and obj_id_int in self.memoria_identidades:
                        nombre_vip, clase_vip = self.memoria_identidades[obj_id_int]
                        obj_id_int = self.registro_biometrico_fijo[nombre_vip] # Magia Gozu
                        
                        if clase_vip == "SOSPECHOSOS":
                            color_caja = (0, 0, 255)
                            nombre_obj = f"PELIGRO: {nombre_vip}"
                            sospechoso_en_pantalla = True
                        elif clase_vip == "CLIENTES":
                            color_caja = (255, 200, 0)
                            nombre_obj = f"CLIENTE: {nombre_vip}"
                        else:
                            color_caja = (0, 255, 0)
                            nombre_obj = f"STAFF: {nombre_vip}"
                            
                    objetos_en_pantalla += 1
                    
                    # Generar registros visuales
                    if self.frame_count % 5 == 0:
                        hex_code = hex(obj_id_int * 7919)[2:].upper().zfill(4)
                        self.hacker_log.append(f"> [{obj_id_int:02d}] {nombre_obj[:9]} {int(conf*100)}% 0x{hex_code}")
                        if len(self.hacker_log) > 22:
                            self.hacker_log.pop(0)
                    
                    # Actualizar Radar
                    h_f, w_f = frame.shape[:2]
                    offset_x = (((x1 + x2) / 2) / w_f) - 0.5 
                    self.odometria.registrar_objeto_en_radar(f"{nombre_obj} ({obj_id_int})", offset_x)
                    
                    # Guardar foto/csv si es de interés
                    if cls_int in [0, 24, 67]: 
                        self.evidencias.registrar_hallazgo(frame_raw, nombre_obj, obj_id_int, conf, self.alertas)
                            
                elif conf >= 0.40:
                    nombre_obj = "OBJ. DESCONOCIDO"
                    color_caja = (100, 100, 100)
                    objetos_en_pantalla += 1
                else:
                    continue 
                
                cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color_caja, 2)
                cv2.putText(frame_anotado, f"[{obj_id_int}] {nombre_obj} {int(conf*100)}%", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja, 2)
                
        return frame_anotado, sospechoso_en_pantalla, objetos_en_pantalla
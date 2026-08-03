import cv2
import numpy as np
import math
import time

class OdometriaORB:
    def __init__(self):
        # Usamos 1000 features para tener mejor estabilidad al calcular la rotación en 3D
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_kp = None
        self.prev_desc = None
        
        # --- DISEÑO DEL MAPA BASE ---
        self.map_size = 600
        self.trayectoria = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.trayectoria[:] = (10, 20, 10) 
        
        # Dibujar Cuadrícula de fondo
        for i in range(0, self.map_size, 50):
            cv2.line(self.trayectoria, (i, 0), (i, self.map_size), (0, 40, 0), 1)
            cv2.line(self.trayectoria, (0, i), (self.map_size, i), (0, 40, 0), 1)
            
        # Posición inicial en el radar (px)
        self.x = 300
        self.y = 300  
        
        # --- VARIABLES DEL MUNDO 3D REAL ---
        # Matriz de Rotación e Identidad (Posición inicial del mundo)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)
        
        self.focal = 800.0
        self.pp = (400, 300)
        
        self.distancia_metros = 0.0
        self.puntos_objetos = [] # Guardará las anomalias encontradas

        # --- TELEMETRÍA DE FLUJO ÓPTICO (navegación SIN YOLO) ---
        # Se calcula a partir de los mismos puntos ORB ya rastreados.
        # flujo_izq/centro/der : velocidad media de los puntos en cada zona (px/frame)
        # divergencia          : expansión radial media (indica acercamiento frontal)
        # flujo_global         : magnitud media global (sirve para saber si nos movemos)
        # muestras             : cuántos puntos válidos respaldan la medición
        self.telemetria_flujo = {
            "izquierda": 0.0, "centro": 0.0, "derecha": 0.0,
            "divergencia": 0.0, "flujo_global": 0.0, "muestras": 0
        }
        
    def registrar_objeto_en_radar(self, nombre_objeto, offset_x=0):
        """Proyecta el objeto alrededor del robot en píxeles para que siempre se vea en el radar"""
        # Calcular vectores en PÍXELES relativos a la orientación actual
        vec_frente = self.cur_R.dot(np.array([[0], [0], [20.0]])) # 20 píxeles hacia adelante
        vec_lado = self.cur_R.dot(np.array([[offset_x * 50.0], [0], [0]])) # desplazamiento lateral en píxeles
        
        # Posicionar relativo a self.x y self.y (las coordenadas reales en el radar)
        obj_x = self.x + int(vec_frente[0][0]) + int(vec_lado[0][0])
        obj_y = self.y - int(vec_frente[2][0]) - int(vec_lado[2][0])
        
        # Limitar a la pantalla para que nunca desaparezca
        obj_x = max(20, min(self.map_size - 20, obj_x))
        obj_y = max(20, min(self.map_size - 20, obj_y))
        
        # Si el objeto ya existe, actualizar
        for obj in self.puntos_objetos:
            if obj["nombre"] == nombre_objeto:
                obj["pos"] = (obj_x, obj_y)
                obj["tiempo"] = time.time()
                return
                
        # Si es nuevo, agregarlo
        self.puntos_objetos.append({"pos": (obj_x, obj_y), "nombre": nombre_objeto, "tiempo": time.time()})

    def _analizar_flujo_optico(self, pts1, pts2, frame_shape):
        """
        Calcula la telemetría de navegación a partir del movimiento de los puntos ORB.

        Fundamento físico (paralaje de movimiento): al desplazarse la cámara, los
        puntos que pertenecen a superficies CERCANAS se desplazan más rápido en la
        imagen que los de superficies lejanas. Esto permite detectar obstáculos
        SIN necesidad de reconocer qué son (funciona con paredes, columnas, cajas...).

        Calcula dos indicadores:
        1) Flujo por zona (izq/centro/der): dónde está lo más cercano.
        2) Divergencia radial: si los puntos se EXPANDEN desde el centro de la imagen,
           significa que nos acercamos frontalmente a una superficie (efecto 'looming').
        """
        h, w = frame_shape[:2]
        cx, cy = w / 2.0, h / 2.0

        desplaz = pts2 - pts1
        magnitudes = np.linalg.norm(desplaz, axis=1)

        # Filtrar emparejamientos absurdos (errores de matching): descartamos saltos
        # mayores al 25% del ancho de la imagen, imposibles entre dos frames seguidos.
        limite = w * 0.25
        validos = magnitudes < limite
        if np.count_nonzero(validos) < 6:
            self._resetear_flujo()
            return

        pts2_v = pts2[validos]
        desplaz_v = desplaz[validos]
        magnitudes_v = magnitudes[validos]

        # --- 1. FLUJO POR ZONA (mediana: resistente a puntos ruidosos) ---
        x = pts2_v[:, 0]
        mask_izq = x < w * 0.33
        mask_der = x > w * 0.66
        mask_cen = ~(mask_izq | mask_der)

        def _mediana(mask):
            return float(np.median(magnitudes_v[mask])) if np.count_nonzero(mask) >= 3 else 0.0

        flujo_izq = _mediana(mask_izq)
        flujo_cen = _mediana(mask_cen)
        flujo_der = _mediana(mask_der)

        # --- 2. DIVERGENCIA RADIAL (detección de acercamiento frontal) ---
        # Para cada punto: ¿se está alejando del centro de la imagen? Si en promedio
        # sí, la escena se está "expandiendo" => nos acercamos a algo de frente.
        radial = pts2_v - np.array([cx, cy], dtype=np.float32)
        radios = np.linalg.norm(radial, axis=1)
        utiles = radios > 20.0  # puntos muy al centro no aportan info radial fiable

        if np.count_nonzero(utiles) >= 5:
            dir_radial = radial[utiles] / radios[utiles][:, None]
            # Componente del desplazamiento en dirección radial, normalizada por el radio:
            # esto aproxima la tasa de expansión (inverso del tiempo-al-contacto).
            expansion = np.sum(desplaz_v[utiles] * dir_radial, axis=1) / radios[utiles]
            divergencia = float(np.median(expansion))
        else:
            divergencia = 0.0

        self.telemetria_flujo = {
            "izquierda": flujo_izq,
            "centro": flujo_cen,
            "derecha": flujo_der,
            "divergencia": divergencia,
            "flujo_global": float(np.median(magnitudes_v)),
            "muestras": int(np.count_nonzero(validos)),
        }

    def _resetear_flujo(self):
        """Deja la telemetría en cero cuando no hay datos fiables (evita decidir con basura)."""
        self.telemetria_flujo = {
            "izquierda": 0.0, "centro": 0.0, "derecha": 0.0,
            "divergencia": 0.0, "flujo_global": 0.0, "muestras": 0
        }

    def obtener_telemetria_flujo(self):
        """Expone la telemetría de flujo óptico al motor de decisión (navegación sin YOLO)."""
        return self.telemetria_flujo

    def procesar_frame(self, frame):
        draw_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # La telemetría se limpia cada frame; solo se rellena si hay datos fiables.
        # Así el motor de decisión nunca navega con información vieja.
        self._resetear_flujo()
        
        kp, desc = self.orb.detectAndCompute(gray, None)
        
        if self.prev_desc is not None and desc is not None and len(kp) > 0 and len(self.prev_kp) > 0:
            matches = self.bf.match(self.prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 10:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

                # --- ANÁLISIS DE FLUJO ÓPTICO (base de la navegación autónoma) ---
                # Se calcula ANTES de la matriz esencial, para que siga funcionando
                # aunque el cálculo de pose 3D falle. No depende de YOLO en absoluto.
                self._analizar_flujo_optico(pts1, pts2, frame.shape)
                
                E, mask = cv2.findEssentialMat(pts2, pts1, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None and E.shape == (3, 3):
                    _, R, t, mask = cv2.recoverPose(E, pts2, pts1, focal=self.focal, pp=self.pp)
                    
                    # --- MATEMÁTICAS AVANZADAS (FÍSICA E INERCIA) ---
                    # 1. Calcular la Pseudo-escala basada en Flujo Óptico
                    # Esto evita que el robot "alucine" que se mueve cuando está quieto
                    inlier_pts1 = []
                    inlier_pts2 = []
                    for i, inlier in enumerate(mask):
                        if inlier:
                            pt = pts2[i]
                            inlier_pts2.append(pt)
                            inlier_pts1.append(pts1[i])
                            cv2.circle(draw_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                            
                    if len(inlier_pts1) > 5:
                        disp = np.linalg.norm(np.array(inlier_pts1) - np.array(inlier_pts2), axis=1)
                        median_disp = np.median(disp)
                    else:
                        median_disp = 0
                        
                    # --- MATEMÁTICAS BRUTALES: FUSIÓN SENSORIAL (VO + IMU VIRTUAL) ---
                    if median_disp > 1.5:
                        # 1. IMU VIRTUAL: GIROSCOPIO (Flujo Óptico)
                        dx_pixels = np.median([p2[0] - p1[0] for p1, p2 in zip(inlier_pts1, inlier_pts2)])
                        yaw_imu = math.atan2(-dx_pixels, self.focal) 
                        
                        # 2. ODOMETRÍA VISUAL (Matriz)
                        yaw_vo = math.atan2(R[0, 2], R[0, 0])
                        
                        # 3. FILTRO DE FUSIÓN DE SENSORES
                        yaw_fusion = (0.85 * yaw_imu) + (0.15 * yaw_vo)
                        yaw_fusion = max(-0.15, min(0.15, yaw_fusion))
                        
                        R_fusion = np.array([
                            [math.cos(yaw_fusion), 0, math.sin(yaw_fusion)],
                            [0, 1, 0],
                            [-math.sin(yaw_fusion), 0, math.cos(yaw_fusion)]
                        ])
                        
                        # 4. IMU VIRTUAL: ACELERÓMETRO Y CINEMÁTICA
                        penalizacion_curva = math.cos(yaw_fusion) ** 4 
                        scale = median_disp * 0.03 * penalizacion_curva
                        
                        vector_impulso = scale * self.cur_R.dot(t)
                        
                        if not hasattr(self, 'velocidad_previa'):
                            self.velocidad_previa = vector_impulso
                        else:
                            masa = 0.85
                            self.velocidad_previa = (1.0 - masa) * vector_impulso + masa * self.velocidad_previa
                            
                        self.cur_t = self.cur_t + self.velocidad_previa
                        self.cur_R = R_fusion.dot(self.cur_R)
                        
                        map_scale = 3.0
                        nueva_x = 300 + int(self.cur_t[0][0] * map_scale)
                        nueva_y = 300 - int(self.cur_t[2][0] * map_scale) 
                        
                        desplazamiento = math.sqrt(self.velocidad_previa[0][0]**2 + self.velocidad_previa[1][0]**2 + self.velocidad_previa[2][0]**2)
                        self.distancia_metros += desplazamiento * 0.15 
                        
                        nueva_x = max(10, min(self.map_size - 10, nueva_x))
                        nueva_y = max(10, min(self.map_size - 10, nueva_y))
                        
                        if (self.x != nueva_x) or (self.y != nueva_y):
                            cv2.line(self.trayectoria, (self.x, self.y), (nueva_x, nueva_y), (0, 0, 255), 2)
                            
                        self.x, self.y = nueva_x, nueva_y

        self.prev_kp = kp
        self.prev_desc = desc
        
        display_map = self.trayectoria.copy()
        
        # --- DIBUJAR PUNTOS DE INTERES EN EL MAPA (EFECTO FANTASMA DE 5 SEGUNDOS) ---
        tiempo_actual = time.time()
        objetos_activos = []
        
        for obj in self.puntos_objetos:
            edad = tiempo_actual - obj["tiempo"]
            if edad <= 5.0:
                pos = obj["pos"]
                # Letras más grandes, blancas puras, con borde negro gordo (Máxima legibilidad)
                texto = obj["nombre"]
                x_texto = pos[0] - 25 # Centrar un poco mejor
                y_texto = pos[1]
                
                # Sombra/Borde oscuro grueso
                cv2.putText(display_map, texto, (x_texto, y_texto), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 4)
                # Texto Blanco Brillante principal
                cv2.putText(display_map, texto, (x_texto, y_texto), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
                objetos_activos.append(obj)
                
        # Limpiar los que ya expiraron
        self.puntos_objetos = objetos_activos
        
        # --- DIBUJAR AL ROBOT CON ORIENTACIÓN (Láser Direccional) ---
        # Calculamos hacia donde apunta el robot (Multiplicando la matriz por un vector hacia el frente en Z)
        vec_frente = self.cur_R.dot(np.array([[0], [0], [15.0]])) # 15 pixeles de largo
        punta_x = self.x + int(vec_frente[0][0])
        punta_y = self.y - int(vec_frente[2][0])
        
        cv2.line(display_map, (self.x, self.y), (punta_x, punta_y), (0, 255, 255), 2) # Laser amarillo
        cv2.circle(display_map, (self.x, self.y), 6, (0, 255, 255), -1) # Cuerpo del robot
        
        # HUD Odometro
        cv2.putText(display_map, f"DISTANCIA: {self.distancia_metros:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return draw_frame, display_map
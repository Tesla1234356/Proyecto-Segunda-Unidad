"""
Módulo de Decisión Autónoma (Capa de Control de Sentinel AI)
================================================================
Convierte la PERCEPCIÓN del sistema en una ACCIÓN de navegación concreta.
Es el eslabón que transforma un sistema que "observa" en uno que "decide".

ARQUITECTURA DE DOS CAPAS
-------------------------
El sistema NO depende de YOLO para navegar. Se apoya en dos señales de
naturaleza distinta y complementaria:

  CAPA 1 - NAVEGACIÓN GEOMÉTRICA (flujo óptico, sin YOLO)
    Responde a: "¿por dónde puedo pasar?"
    Detecta CUALQUIER superficie cercana (paredes, columnas, muebles sin
    clasificar) mediante el paralaje de movimiento de los puntos ORB que
    la odometría ya rastrea. No necesita saber qué es el obstáculo.

  CAPA 2 - SEMÁNTICA (YOLOv8, opcional)
    Responde a: "¿qué es lo que veo y cuánto me importa?"
    Aporta contexto de seguridad (un sospechoso obliga a detenerse) y
    matiza el riesgo (un mueble no se aparta; una persona sí).

La navegación funciona aunque YOLO no detecte absolutamente nada.
YOLO enriquece la decisión, no la sostiene.

TÉCNICAS APLICADAS EN LA CAPA 1
-------------------------------
1) Estrategia de balance de flujo (inspirada en la navegación de insectos):
   se compara la velocidad del flujo del lado izquierdo contra el derecho.
   El lado con MENOS flujo es el que tiene el espacio más despejado.
   Al ser una comparación RELATIVA, es robusta frente a cambios de
   velocidad de marcha, resolución o iluminación.

2) Divergencia radial / 'looming': si los puntos se expanden desde el
   centro de la imagen, la cámara se aproxima de frente a una superficie.
   Éste es el indicador que detecta una PARED LISA, invisible para YOLO.

3) Compuerta de ego-movimiento: si la cámara está prácticamente quieta,
   el flujo no contiene información de profundidad y el sistema se
   abstiene de ordenar maniobras (evita decisiones sobre ruido).
"""

from collections import deque

# ---------------------------------------------------------------------------
# CAPA SEMÁNTICA (YOLO) - pesos por tipo de objeto
# ---------------------------------------------------------------------------
# Clases COCO consideradas obstáculos estáticos "duros" (no se apartan solos)
# 13: banco | 56: silla | 57: sofá | 58: planta | 59: cama | 60: mesa
OBSTACULOS_ESTATICOS = {13, 56, 57, 58, 59, 60}

PESO_OBSTACULO_DURO = 1.0   # mobiliario: bloquea de verdad
PESO_PERSONA = 0.45         # una persona se aparta o puede rodearse
PESO_DEFECTO = 0.70         # resto de objetos (mochilas, cajas, etc.)
CONF_MINIMA_YOLO = 0.40     # mismo criterio de "objeto real" que usa core.py

# ---------------------------------------------------------------------------
# CAPA GEOMÉTRICA (flujo óptico) - parámetros de navegación
# ---------------------------------------------------------------------------
# Movimiento mínimo (px/frame) para considerar que la cámara avanza.
# Por debajo de esto el flujo es ruido y no aporta profundidad.
FLUJO_MINIMO_EGOMOVIMIENTO = 1.2

# Muestras mínimas de puntos ORB para fiarse de la telemetría.
MUESTRAS_MINIMAS = 8

# Desbalance lateral (0..1) a partir del cual un lado se considera obstruido.
# Se calcula como |izq - der| / (izq + der): es una proporción, no un valor absoluto.
UMBRAL_DESBALANCE = 0.28

# Divergencia radial a partir de la cual se considera aproximación frontal.
# Es una tasa de expansión por frame; valor pequeño y adimensional.
UMBRAL_DIVERGENCIA = 0.020
UMBRAL_DIVERGENCIA_CRITICA = 0.045   # impacto inminente: retroceder

# Riesgo de zona (0..1) a partir del cual se considera bloqueada.
UMBRAL_BLOQUEO = 0.35

# Frames consecutivos que debe repetirse una decisión antes de aceptarla
# (evita que el HUD parpadee entre órdenes por ruido de un solo frame).
FRAMES_ESTABILIDAD = 4


class MotorDecision:
    """Capa de control autónomo: percepción -> decisión de navegación."""

    def __init__(self):
        self.candidato = None
        self.contador_candidato = 0
        self.accion_estable = "EN ESPERA // SIN MOVIMIENTO"
        self.color_estable = (200, 200, 200)
        self.fuente_estable = "-"
        self.historial = deque(maxlen=60)

    # =====================================================================
    # CAPA 2 - SEMÁNTICA (YOLO)
    # =====================================================================
    def _zona_de(self, centro_x, ancho):
        if centro_x < ancho * 0.33:
            return "izquierda"
        if centro_x > ancho * 0.66:
            return "derecha"
        return "centro"

    def _peso_clase(self, cls_int):
        if cls_int in OBSTACULOS_ESTATICOS:
            return PESO_OBSTACULO_DURO
        if cls_int == 0:
            return PESO_PERSONA
        return PESO_DEFECTO

    def _riesgo_semantico(self, boxes, classes, confs, ancho, alto):
        """Riesgo por zona según objetos reconocidos. Cercanía ~ área de la caja."""
        riesgo = {"izquierda": 0.0, "centro": 0.0, "derecha": 0.0}
        area_frame = float(ancho * alto) or 1.0

        for box, cls, conf in zip(boxes, classes, confs):
            if conf < CONF_MINIMA_YOLO:
                continue
            x1, y1, x2, y2 = box
            proximidad = (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / area_frame
            zona = self._zona_de((x1 + x2) / 2.0, ancho)
            # Escalado x3: un objeto que ocupa ~33% del frame satura la zona.
            riesgo[zona] = min(1.0, riesgo[zona] + proximidad * self._peso_clase(int(cls)) * 3.0)

        return riesgo

    # =====================================================================
    # CAPA 1 - GEOMÉTRICA (flujo óptico, sin YOLO)
    # =====================================================================
    def _riesgo_geometrico(self, tele):
        """
        Traduce la telemetría de flujo en riesgo por zona (0..1).
        Devuelve (riesgo, hay_datos, divergencia).
        """
        riesgo = {"izquierda": 0.0, "centro": 0.0, "derecha": 0.0}

        if not tele or tele.get("muestras", 0) < MUESTRAS_MINIMAS:
            return riesgo, False, 0.0

        # Compuerta de ego-movimiento: sin desplazamiento no hay profundidad medible.
        if tele.get("flujo_global", 0.0) < FLUJO_MINIMO_EGOMOVIMIENTO:
            return riesgo, False, 0.0

        izq = tele.get("izquierda", 0.0)
        cen = tele.get("centro", 0.0)
        der = tele.get("derecha", 0.0)
        divergencia = tele.get("divergencia", 0.0)

        # --- Balance lateral (comparación relativa, no absoluta) ---
        suma_lat = izq + der
        if suma_lat > 1e-6:
            desbalance = (izq - der) / suma_lat  # >0: izquierda más rápida => más cerca
            if desbalance > 0:
                riesgo["izquierda"] = min(1.0, desbalance / UMBRAL_DESBALANCE * UMBRAL_BLOQUEO)
            else:
                riesgo["derecha"] = min(1.0, -desbalance / UMBRAL_DESBALANCE * UMBRAL_BLOQUEO)

        # --- Aproximación frontal por divergencia radial ---
        # Éste es el mecanismo que detecta una pared lisa sin ayuda de YOLO.
        if divergencia > 0:
            riesgo["centro"] = min(1.0, divergencia / UMBRAL_DIVERGENCIA * UMBRAL_BLOQUEO)

        # Refuerzo: si además el flujo central supera claramente al lateral,
        # hay una superficie frontal aunque la divergencia sea moderada.
        media_lat = (izq + der) / 2.0
        if media_lat > 1e-6 and cen > media_lat * 1.6:
            riesgo["centro"] = max(riesgo["centro"], UMBRAL_BLOQUEO * 1.05)

        # --- PRUEBA DE ESCAPE ---
        # El balance lateral es RELATIVO: si ambos lados están igual de cerca,
        # el desbalance es nulo y ambos parecerían despejados (caso callejón sin
        # salida). Para evitarlo, cuando el frente está bloqueado se comprueba si
        # cada lado ofrece un flujo SUSTANCIALMENTE menor que el frontal. Un lado
        # con flujo comparable al del frente no es una vía de escape.
        if cen > 1e-6 and riesgo["centro"] >= UMBRAL_BLOQUEO:
            for zona, valor in (("izquierda", izq), ("derecha", der)):
                if valor / cen >= 0.75:
                    riesgo[zona] = max(riesgo[zona], UMBRAL_BLOQUEO * 1.05)

        return riesgo, True, divergencia

    # =====================================================================
    # FUSIÓN Y DECISIÓN
    # =====================================================================
    def _fusionar(self, geo, sem):
        """Riesgo final por zona: el mayor de ambas capas (criterio conservador)."""
        return {z: max(geo.get(z, 0.0), sem.get(z, 0.0)) for z in ("izquierda", "centro", "derecha")}

    def _decidir_cruda(self, riesgo, divergencia, hay_flujo, sospechoso):
        """Decisión del frame actual, sin filtrar. Devuelve (accion, color, fuente)."""
        # PRIORIDAD 1 - Seguridad: un sospechoso detiene la patrulla.
        if sospechoso:
            return "DETENER // INTRUSO DETECTADO", (0, 0, 255), "SEMANTICA"

        # PRIORIDAD 2 - Impacto frontal inminente.
        if divergencia > UMBRAL_DIVERGENCIA_CRITICA:
            return "RETROCEDER // OBSTACULO INMINENTE", (0, 0, 255), "FLUJO"

        izq, cen, der = riesgo["izquierda"], riesgo["centro"], riesgo["derecha"]

        # PRIORIDAD 3 - Camino libre al frente.
        if cen < UMBRAL_BLOQUEO:
            if not hay_flujo:
                return "EN ESPERA // SIN MOVIMIENTO", (200, 200, 200), "-"

            # Corrección de centrado: aunque el frente esté despejado, si un lado
            # está claramente más cerca, el robot se aparta de él para mantenerse
            # centrado en el pasillo (comportamiento propio de la estrategia de
            # balance de flujo). Es una corrección suave, no una maniobra evasiva.
            margen_correccion = UMBRAL_BLOQUEO * 0.75
            if izq > margen_correccion and izq > der * 1.5:
                return "AVANZAR // CORRIGIENDO DERECHA", (0, 255, 180), "FLUJO"
            if der > margen_correccion and der > izq * 1.5:
                return "AVANZAR // CORRIGIENDO IZQUIERDA", (0, 255, 180), "FLUJO"

            return "AVANZAR", (0, 255, 100), "FLUJO+YOLO"

        # PRIORIDAD 4 - Frente bloqueado: buscar el lado más despejado.
        if izq >= UMBRAL_BLOQUEO and der >= UMBRAL_BLOQUEO:
            return "RETROCEDER // SIN SALIDA", (0, 0, 255), "FLUJO+YOLO"

        if izq < der:
            return "GIRAR IZQUIERDA", (0, 220, 255), "FLUJO+YOLO"
        return "GIRAR DERECHA", (0, 220, 255), "FLUJO+YOLO"

    # =====================================================================
    # API PÚBLICA
    # =====================================================================
    def decidir_accion(self, resultados_yolo, sospechoso_detectado, ancho_frame,
                       alto_frame, telemetria_flujo=None):
        """
        Punto de entrada. Combina la telemetría de flujo (navegación) con las
        detecciones de YOLO (semántica) y devuelve una orden ESTABLE.

        Retorna: (accion, color_bgr, detalle_texto)
        """
        # --- Capa geométrica (funciona sin YOLO) ---
        riesgo_geo, hay_flujo, divergencia = self._riesgo_geometrico(telemetria_flujo)

        # --- Capa semántica (si YOLO aportó detecciones) ---
        boxes, classes, confs = [], [], []
        if resultados_yolo is not None and resultados_yolo[0].boxes.id is not None:
            boxes = resultados_yolo[0].boxes.xyxy.cpu().numpy()
            classes = resultados_yolo[0].boxes.cls.cpu().numpy()
            confs = resultados_yolo[0].boxes.conf.cpu().numpy()
        riesgo_sem = self._riesgo_semantico(boxes, classes, confs, ancho_frame, alto_frame)

        # --- Fusión y decisión ---
        riesgo = self._fusionar(riesgo_geo, riesgo_sem)
        accion, color, fuente = self._decidir_cruda(
            riesgo, divergencia, hay_flujo, sospechoso_detectado
        )

        # --- Filtro de estabilidad (anti-parpadeo) ---
        if accion == self.candidato:
            self.contador_candidato += 1
        else:
            self.candidato = accion
            self.contador_candidato = 1

        critica = "INTRUSO" in accion or "INMINENTE" in accion or "SIN SALIDA" in accion
        if critica or self.contador_candidato >= FRAMES_ESTABILIDAD:
            self.accion_estable = accion
            self.color_estable = color
            self.fuente_estable = fuente

        self.historial.append(self.accion_estable)

        detalle = (f"RIESGO I:{riesgo['izquierda']*100:.0f} "
                   f"C:{riesgo['centro']*100:.0f} "
                   f"D:{riesgo['derecha']*100:.0f} | "
                   f"DIV:{divergencia:+.3f} | {self.fuente_estable}")

        return self.accion_estable, self.color_estable, detalle
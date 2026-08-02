import cv2
import numpy as np

def draw_tech_corners(img, pt1, pt2, color, length=20, thickness=2):
    """Dibuja esquinas tecnológicas estilo HUD de Iron Man"""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1, y1), (x1+length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1+length), color, thickness)
    cv2.line(img, (x2, y1), (x2-length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1+length), color, thickness)
    cv2.line(img, (x1, y2), (x1+length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2-length), color, thickness)
    cv2.line(img, (x2, y2), (x2-length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2-length), color, thickness)

def crear_hud_base():
    """Genera la plantilla estática del HUD futurista."""
    # Ampliamos el canvas a 1550 para tener espacio para la 3ra columna (Terminal Hacker)
    hud = np.zeros((720, 1550, 3), dtype=np.uint8)
    
    # Fondo Oscuro Profundo (Estilo J.A.R.V.I.S)
    hud[:] = (18, 12, 10) 
    
    # Grid Punteado sutil en vez de rayas feas
    for i in range(0, 1550, 30):
        for j in range(0, 720, 30):
            hud[j, i] = (60, 40, 30) 
            
    # --- BORDES Y TITULOS ESTATICOS (COLORES NEÓN) ---
    # Panel Principal de Cámara (Bordes Cyan brillante)
    cv2.rectangle(hud, (40, 60), (840, 660), (30, 20, 15), -1) 
    draw_tech_corners(hud, (35, 55), (845, 665), (255, 220, 0), length=30, thickness=3) 
    cv2.putText(hud, "SYSTEM OVERRIDE // VISION OPTICA", (40, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 220, 0), 1)
    
    # Panel Radar (Bordes Verde Militar)
    cv2.rectangle(hud, (880, 60), (1230, 410), (15, 25, 15), -1)
    draw_tech_corners(hud, (875, 55), (1235, 415), (0, 255, 100), length=25, thickness=2) 
    cv2.putText(hud, "SENSOR RADAR L.I.D.A.R. VIRTUAL", (880, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 100), 1)
    
    # Panel Diagnostico (Bordes Magenta/Morado Cyberpunk)
    cv2.rectangle(hud, (880, 440), (1230, 660), (15, 10, 15), -1)
    draw_tech_corners(hud, (875, 435), (1235, 665), (255, 50, 200), length=20, thickness=2) 
    cv2.putText(hud, "> PANEL DE DIAGNOSTICO CORE <", (895, 465), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 50, 200), 1)
    
    # NUEVO: Panel Vertical Hacker (Extrema Derecha)
    cv2.rectangle(hud, (1260, 60), (1520, 660), (10, 20, 10), -1)
    draw_tech_corners(hud, (1255, 55), (1525, 665), (0, 255, 0), length=25, thickness=2) 
    cv2.putText(hud, "TARGET LOCK TERMINAL", (1260, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)
    
    # Título Global
    cv2.putText(hud, "SENTINEL AI PROTOCOL V3.0", (40, 25), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)
    cv2.putText(hud, "EVIDENCIAS LOG:", (895, 570), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (150, 150, 150), 1)
    
    return hud

def actualizar_hud(hud, frame_principal, radar, fps, objetos_count, alertas, hacker_log):
    """Pega el video manteniendo sus proporciones originales y dibuja textos dinámicos."""
    # 1. Pegar CÁMARA PRINCIPAL sin estirar ni deformar
    h_f, w_f = frame_principal.shape[:2]
    # Calcular coordenadas para centrar el video dentro del marco de 800x600
    x_offset = 40 + (800 - w_f) // 2
    y_offset = 60 + (600 - h_f) // 2
    hud[y_offset:y_offset+h_f, x_offset:x_offset+w_f] = frame_principal
    
    # 2. Pegar RADAR (El radar es cuadrado y lo redimensionamos estricto a 350x350)
    r_resized = cv2.resize(radar, (350, 350))
    hud[60:410, 880:1230] = r_resized
    
    # 3. Dibujar Datos Dinámicos
    color_fps = (0, 255, 100) if fps > 10 else (0, 0, 255)
    cv2.putText(hud, f"FPS REALES   : {fps:.1f}", (895, 510), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color_fps, 1)
    cv2.putText(hud, f"OBJETOS SCAN : {objetos_count}", (895, 540), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 220, 0), 1)
    
    # Imprimir historial de alertas recientes
    y_alertas = 590
    for alerta in alertas[-3:]:
        cv2.putText(hud, alerta, (895, y_alertas), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 50, 200), 1)
        y_alertas += 22
        
    # --- EFECTO HACKER (TERMINAL SCROLLING) ---
    cv2.putText(hud, "LIVE TRACKING FEED:", (1270, 90), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
    y_hacker = 130
    for tgt in hacker_log:
        cv2.putText(hud, tgt, (1270, y_hacker), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
        cv2.line(hud, (1270, y_hacker+5), (1510, y_hacker+5), (0, 40, 0), 1) 
        y_hacker += 24
            
    return hud

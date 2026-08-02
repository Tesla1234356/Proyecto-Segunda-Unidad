import cv2
import os
from modules.reconocimiento import ReconocimientoFacial

def probar_reconocimiento():
    print("Iniciando Etapa 3: Reconocimiento Facial con FaceNet...")
    print("-------------------------------------------------------")
    print("Por favor, asegúrate de haber puesto al menos 1 foto tuya en:")
    print("-> data/rostros/TuNombre.jpg")
    print("-------------------------------------------------------")
    
    # Inicializar la Inteligencia Artificial de Rostros
    reconocedor = ReconocimientoFacial()

    # Para esta prueba, encendemos la CÁMARA WEB de tu laptop (0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo encender la cámara web.")
        return

    # Reducimos los FPS artificialmente porque DeepFace procesando 
    # todos los frames seguidos pone lenta la computadora
    frame_count = 0
    # Variable para recordar la última cara vista y no parpadear
    ultimo_rostro = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Procesamos 1 de cada 5 fotogramas para que no haya lag extremo
        if frame_count % 5 == 0:
            nuevo_rostro = reconocedor.reconocer_frame(frame)
            if nuevo_rostro is not None:
                ultimo_rostro = nuevo_rostro
                
        # DIBUJAR SIEMPRE (incluso en los frames donde no procesamos)
        draw_frame = frame.copy()
        if ultimo_rostro is not None:
            nombre, x, y, w, h = ultimo_rostro
            cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(draw_frame, f"VIP: {nombre}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Escáner VIP (Etapa 3)", draw_frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Etapa 3 Finalizada.")

if __name__ == '__main__':
    probar_reconocimiento()

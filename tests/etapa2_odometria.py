import cv2
from modules.odometria import OdometriaORB

def probar_odometria(video_path):
    print("Iniciando Etapa 2: Odometría Visual con ORB...")
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir el video.")
        return

    # Inicializar nuestra clase de Odometría
    odometria = OdometriaORB()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame = cv2.resize(frame, (800, 600))

            # Procesar el frame con nuestra lógica de Odometría
            # Retorna el frame con los puntitos verdes dibujados, y un mapa negro con el recorrido
            frame_puntos, mapa_trayectoria = odometria.procesar_frame(frame)

            # Mostrar las ventanas
            cv2.imshow("Vista de la Camara (Puntos ORB)", frame_puntos)
            cv2.imshow("Mapa del Robot (Trayectoria)", mapa_trayectoria)

            # Para que el video no vaya súper rápido como te pasó antes,
            # Le ponemos un "delay" de 30 milisegundos (Aprox. 30 FPS normales)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Etapa 2 Finalizada.")

if __name__ == '__main__':
    probar_odometria("data/video_movimiento2.mp4")

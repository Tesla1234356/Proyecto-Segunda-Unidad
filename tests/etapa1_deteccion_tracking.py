import cv2
from ultralytics import YOLO

def probar_deteccion_tracking(video_path):
    print("Iniciando Etapa 1: Detección y Tracking con YOLOv8...")
    
    # Cargamos el modelo YOLOv8 versión Nano (el más ligero y rápido)
    model = YOLO('yolov8n.pt') 

    # Abrir el video
    cap = cv2.VideoCapture("data/video_prueba.mp4")
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video en {video_path}")
        print("Por favor, asegúrate de colocar un video llamado 'video_prueba.mp4' en la carpeta.")
        return

    while cap.isOpened():
        # Leer un fotograma del video
        success, frame = cap.read()

        if success:
            # Redimensionar para que la ventana no sea gigante y procese más rápido
            frame = cv2.resize(frame, (800, 600))

            # Ejecutar Detección y Tracking (usando ByteTrack interno de YOLOv8)
            # classes=[0] significa que SOLO detectará personas (ignora carros, perros, etc.)
            results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)

            # Dibujar las cajas y los IDs (Tracking) en el fotograma
            annotated_frame = results[0].plot()

            # Mostrar el resultado
            cv2.imshow("Etapa 1: Deteccion y Tracking (Sentinel AI)", annotated_frame)

            # Presiona 'q' para salir del video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Fin del video
            break

    # Liberar la memoria
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Etapa 1 Finalizada con éxito.")

if __name__ == '__main__':
    # Asegúrate de tener el video_prueba.mp4 en la misma carpeta
    probar_deteccion_tracking("video_prueba.mp4")

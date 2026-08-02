import cv2
from deepface import DeepFace
import os

class ReconocimientoFacial:
    def __init__(self, db_path="data/rostros"):
        self.db_path = db_path
        # Crear la carpeta si no existe
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
    def reconocer_frame(self, frame):
        # En lugar de dibujar, ahora devolvemos los datos para mantenerlos en memoria
        try:
            resultados = DeepFace.find(img_path=frame, 
                                       db_path=self.db_path, 
                                       model_name="Facenet", 
                                       enforce_detection=False, 
                                       silent=True)
            
            if len(resultados) > 0 and len(resultados[0]) > 0:
                df = resultados[0]
                identidad_path = df.iloc[0]['identity']
                
                # Extraer nombre del archivo y la subcarpeta (la clase)
                nombre = os.path.basename(identidad_path).split('.')[0]
                clase = os.path.basename(os.path.dirname(identidad_path)).upper()
                
                # Si las fotos están sueltas en 'rostros' sin subcarpeta, le ponemos DESCONOCIDO
                if clase == "ROSTROS":
                    clase = "AUTORIZADO" # Por defecto si no hicieron carpetas
                
                x = df.iloc[0]['source_x']
                y = df.iloc[0]['source_y']
                w = df.iloc[0]['source_w']
                h = df.iloc[0]['source_h']
                
                # Devolvemos el nombre, la clase y la cajita
                return (nombre, clase, x, y, w, h)
                            
        except Exception as e:
            pass
            
        # Si no hay cara, devolvemos None
        return None

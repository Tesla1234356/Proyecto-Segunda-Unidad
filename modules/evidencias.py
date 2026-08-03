import os
import cv2
import csv
from datetime import datetime

class GestorEvidencias:
    """Gestiona el guardado de fotos y registros CSV (Logs) de los objetos detectados."""
    def __init__(self, ruta_base="evidencias"):
        self.ruta_base = ruta_base
        self.csv_path = os.path.join(self.ruta_base, "historial_objetos.csv")
        self.registros_id = set()
        
        if not os.path.exists(self.ruta_base):
            os.makedirs(self.ruta_base)
            
    def registrar_hallazgo(self, frame_raw, nombre_obj, obj_id_int, conf, alertas):
        """Si el ID es nuevo, guarda su foto y lo anota en la base de datos CSV."""
        if obj_id_int not in self.registros_id:
            fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Guardar captura limpia
            nombre_archivo = nombre_obj.replace(':', '-')
            ruta_img = os.path.join(self.ruta_base, f"EVID_{nombre_archivo}_ID{obj_id_int}_{fecha_hora}.jpg")
            cv2.imwrite(ruta_img, frame_raw)
            
            # Notificar al HUD (Interfaz Hacker)
            alertas.append(f"> LOG: {nombre_obj} detectado (ID:{obj_id_int})")
            
            # Guardar en Base de Datos Excel (CSV)
            archivo_existe = os.path.exists(self.csv_path)
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not archivo_existe:
                    writer.writerow(['Fecha_Hora', 'ID_Tracking', 'Objeto', 'Confianza_Deteccion'])
                writer.writerow([fecha_hora, obj_id_int, nombre_obj, f"{int(conf*100)}%"])
                
            self.registros_id.add(obj_id_int)

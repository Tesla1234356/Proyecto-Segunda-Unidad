import pygame

class GestorAudio:
    """Controla el sistema de alarmas y sonido de la Inteligencia Artificial."""
    def __init__(self, ruta_alarma="data/sonidos/alarma.mp3"):
        pygame.mixer.init()
        self.alarma_cargada = False
        try:
            pygame.mixer.music.load(ruta_alarma)
            self.alarma_cargada = True
        except Exception:
            print("[!] No se encontró el archivo de audio de la alarma en la ruta especificada.")
            
    def procesar_alarma(self, peligro_detectado):
        """Activa la sirena si hay peligro, o la apaga si el peligro desapareció."""
        if not self.alarma_cargada:
            return
            
        if peligro_detectado:
            if not pygame.mixer.music.get_busy(): # Si no está sonando ya...
                pygame.mixer.music.play(-1) # Loop infinito
        else:
            if pygame.mixer.music.get_busy(): # Si el peligro ya no está pero sigue sonando...
                pygame.mixer.music.stop() # La apagamos inmediatamente

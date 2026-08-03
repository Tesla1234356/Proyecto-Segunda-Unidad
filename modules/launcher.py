import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk

def mostrar_launcher(cargador_callback):
    """Muestra un panel hacker con barra de carga y luego permite seleccionar la fuente."""
    fuente = None
    
    root = tk.Tk()
    root.title("Sentinel AI - Boot Sequence")
    root.geometry("450x300")
    root.configure(bg="#0D1117")
    
    # Centrar ventana y evitar que se cambie el tamaño
    root.eval('tk::PlaceWindow . center')
    root.resizable(False, False)
    
    # --- PANTALLA 1: SPLASH SCREEN DE CARGA ---
    lbl_title = tk.Label(root, text="INICIANDO SISTEMA...", fg="#00FF00", bg="#0D1117", font=("Courier", 18, "bold"))
    lbl_title.pack(pady=40)
    
    lbl_status = tk.Label(root, text="Preparando entorno...", fg="white", bg="#0D1117", font=("Courier", 10))
    lbl_status.pack(pady=10)
    
    # Barra de progreso Aestetic
    style = ttk.Style()
    style.theme_use('default')
    style.configure("green.Horizontal.TProgressbar", foreground='#00FF00', background='#00FF00')
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", style="green.Horizontal.TProgressbar")
    progress.pack(pady=10)
    
    # --- PANTALLA 2: MENÚ DE SELECCIÓN ---
    def mostrar_menu_seleccion():
        for widget in root.winfo_children():
            widget.destroy()
            
        tk.Label(root, text="SENTINEL AI PROTOCOL V3.0", fg="#00FF00", bg="#0D1117", font=("Courier", 16, "bold")).pack(pady=20)
        tk.Label(root, text="MODELOS CARGADOS. SELECCIONE ENTRADA:", fg="white", bg="#0D1117", font=("Courier", 10)).pack(pady=10)
        
        def set_video():
            nonlocal fuente
            ruta = filedialog.askopenfilename(title="Cargar Archivo", filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov"), ("Todos", "*.*")])
            if ruta:
                fuente = ruta
                root.quit() # Sale del mainloop para continuar
                
        def set_webcam():
            nonlocal fuente
            fuente = 1
            root.quit()
            
        def set_ipcam():
            nonlocal fuente
            url = simpledialog.askstring("Conexión IP Remota", "Ingresa la URL de la cámara de tu celular\n(ej: http://192.168.1.10:8080/video):")
            if url:
                fuente = url
                root.quit()

        tk.Button(root, text="[1] CARGAR VIDEO LOCAL", command=set_video, bg="#21262D", fg="#58A6FF", font=("Courier", 10, "bold"), width=30, relief="flat").pack(pady=5)
        tk.Button(root, text="[2] CAMARA WEB (PC)", command=set_webcam, bg="#21262D", fg="#3FB950", font=("Courier", 10, "bold"), width=30, relief="flat").pack(pady=5)
        tk.Button(root, text="[3] CAMARA DE CELULAR (IP)", command=set_ipcam, bg="#21262D", fg="#F85149", font=("Courier", 10, "bold"), width=30, relief="flat").pack(pady=5)

    def iniciar_carga():
        # Ejecutamos el callback para cargar YOLO y FaceNet
        cargador_callback(lbl_status, progress, root)
        # Cuando termina, cambiamos la pantalla a los botones
        mostrar_menu_seleccion()

    # Arrancar la carga después de 500ms para que la UI se logre dibujar primero
    root.after(500, iniciar_carga)
    root.mainloop()
    
    # Destruir ventana al finalizar
    root.destroy()
    return fuente

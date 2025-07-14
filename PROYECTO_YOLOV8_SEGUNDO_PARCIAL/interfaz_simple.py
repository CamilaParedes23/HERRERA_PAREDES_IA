#!/usr/bin/env python3
"""
Interfaz Gráfica Simplificada para YOLOv8
Herrera & Paredes - Segundo Parcial IA
Versión de respaldo con funcionalidad básica garantizada
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import os

# Configurar ambiente para evitar errores
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO no disponible, modo demostración activado")

class YOLOv8SimpleGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        
        # Variables
        self.model = None
        self.current_image = None
        self.cap = None
        self.webcam_active = False
        
        # Variables de configuración
        self.confidence = tk.DoubleVar(value=0.5)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.max_detections = tk.IntVar(value=100)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=True)
        
        # Crear interfaz
        self.create_interface()
        
        # Cargar modelo
        self.load_model()
    
    def setup_window(self):
        """Configura la ventana principal"""
        self.root.title("YOLOv8 Detector - Herrera & Paredes")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_interface(self):
        """Crea la interfaz principal"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="🎯 YOLOv8 Object Detection",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(
            main_frame,
            text="Herrera & Paredes - Segundo Parcial IA",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#666666'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Frame de contenido
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Controles
        self.create_controls_panel(content_frame)
        
        # Panel derecho - Visualización
        self.create_display_panel(content_frame)
        
        # Barra de estado
        self.create_status_bar(main_frame)
    
    def create_controls_panel(self, parent):
        """Crea el panel de controles"""
        controls_frame = tk.LabelFrame(
            parent,
            text="Panel de Control",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # Configuración
        config_frame = tk.LabelFrame(
            controls_frame,
            text="Configuración",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        )
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Controles deslizantes
        self.create_slider(config_frame, "Confianza:", self.confidence, 0.1, 1.0)
        self.create_slider(config_frame, "IoU Threshold:", self.iou_threshold, 0.1, 1.0)
        self.create_slider(config_frame, "Max Detecciones:", self.max_detections, 10, 300)
        
        # Checkboxes
        tk.Checkbutton(
            config_frame,
            text="Mostrar etiquetas",
            variable=self.show_labels,
            bg='#f0f0f0'
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        tk.Checkbutton(
            config_frame,
            text="Mostrar confianza",
            variable=self.show_confidence,
            bg='#f0f0f0'
        ).pack(anchor=tk.W, padx=5, pady=2)
        
        # Acciones
        actions_frame = tk.LabelFrame(
            controls_frame,
            text="Acciones",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        )
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botones
        tk.Button(
            actions_frame,
            text="📁 Cargar Imagen",
            command=self.load_image,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(
            actions_frame,
            text="🎨 Imagen de Prueba",
            command=self.create_test_image,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(
            actions_frame,
            text="🚀 Detectar Objetos",
            command=self.run_detection,
            bg='#FF9800',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(fill=tk.X, padx=5, pady=2)
        
        # Webcam
        webcam_frame = tk.LabelFrame(
            controls_frame,
            text="Cámara Web",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        )
        webcam_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.webcam_btn = tk.Button(
            webcam_frame,
            text="📹 Iniciar Cámara",
            command=self.toggle_webcam,
            bg='#F44336',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.webcam_btn.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(
            webcam_frame,
            text="💾 Guardar Frame",
            command=self.save_frame,
            bg='#9C27B0',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(fill=tk.X, padx=5, pady=2)
        
        # Información
        tk.Button(
            controls_frame,
            text="ℹ️ Información",
            command=self.show_info,
            bg='#607D8B',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(fill=tk.X, padx=5, pady=10)
    
    def create_slider(self, parent, label, variable, from_, to):
        """Crea un control deslizante"""
        frame = tk.Frame(parent, bg='#f0f0f0')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(frame, text=label, bg='#f0f0f0').pack(anchor=tk.W)
        
        scale = tk.Scale(
            frame,
            from_=from_, to=to,
            resolution=0.05 if isinstance(variable, tk.DoubleVar) else 10,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg='#f0f0f0'
        )
        scale.pack(fill=tk.X)
    
    def create_display_panel(self, parent):
        """Crea el panel de visualización"""
        display_frame = tk.LabelFrame(
            parent,
            text="Visualización",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(
            display_frame,
            bg='white',
            highlightthickness=1,
            highlightbackground='#cccccc'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mensaje inicial
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Muestra mensaje de bienvenida"""
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 300,
            text="🎯 YOLOv8 Detector Listo\\n\\n📁 Carga una imagen\\n🎨 Genera imagen de prueba\\n📹 Inicia la cámara\\n🚀 Ejecuta detección",
            font=('Arial', 14),
            fill='#666666',
            justify=tk.CENTER
        )
    
    def create_status_bar(self, parent):
        """Crea la barra de estado"""
        status_frame = tk.Frame(parent, bg='#e0e0e0', height=30)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="✅ Sistema listo",
            bg='#e0e0e0',
            fg='#333333',
            font=('Arial', 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.fps_label = tk.Label(
            status_frame,
            text="",
            bg='#e0e0e0',
            fg='#666666',
            font=('Arial', 10)
        )
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def load_model(self):
        """Carga el modelo YOLOv8"""
        try:
            if YOLO_AVAILABLE:
                self.model = YOLO('yolov8n.pt')
                self.status_label.config(text="✅ Modelo YOLOv8 cargado")
            else:
                self.status_label.config(text="⚠️ Modo demostración (YOLO no disponible)")
        except Exception as e:
            self.status_label.config(text=f"❌ Error cargando modelo: {str(e)}")
    
    def load_image(self):
        """Carga una imagen desde archivo"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Cargar imagen
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_image = image
                    self.display_image(image)
                    self.status_label.config(text=f"✅ Imagen cargada: {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
    
    def create_test_image(self):
        """Crea una imagen de prueba"""
        try:
            # Crear imagen sintética con formas
            img = np.ones((480, 640, 3), dtype=np.uint8) * 240
            
            # Dibujar algunas formas
            cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), -1)
            cv2.circle(img, (400, 100), 60, (0, 255, 0), -1)
            cv2.rectangle(img, (300, 200), (500, 350), (0, 0, 255), -1)
            
            # Agregar texto
            cv2.putText(img, "Imagen de Prueba", (200, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            self.current_image = img
            self.display_image(img)
            self.status_label.config(text="✅ Imagen de prueba generada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generando imagen: {str(e)}")
    
    def display_image(self, image):
        """Muestra una imagen en el canvas"""
        try:
            # Redimensionar imagen para el canvas
            height, width = image.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            
            # Calcular escala
            scale = min(canvas_width / width, canvas_height / height) * 0.9
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Redimensionar
            resized = cv2.resize(image, (new_width, new_height))
            
            # Convertir a PIL y mostrar
            image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Limpiar canvas y mostrar imagen
            self.canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            print(f"Error mostrando imagen: {e}")
    
    def run_detection(self):
        """Ejecuta la detección de objetos"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            if YOLO_AVAILABLE and self.model:
                # Ejecutar detección real
                results = self.model(self.current_image, 
                                   conf=self.confidence.get(),
                                   iou=self.iou_threshold.get(),
                                   max_det=self.max_detections.get())
                
                # Dibujar resultados
                annotated_image = results[0].plot()
                self.display_image(annotated_image)
                
                # Mostrar estadísticas
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                self.status_label.config(text=f"✅ {num_detections} objetos detectados")
                
            else:
                # Simulación de detección
                result_image = self.current_image.copy()
                
                # Dibujar bounding boxes falsos
                cv2.rectangle(result_image, (100, 100), (300, 200), (0, 255, 0), 2)
                cv2.putText(result_image, "person (0.85)", (100, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.rectangle(result_image, (350, 150), (500, 300), (255, 0, 0), 2)
                cv2.putText(result_image, "car (0.72)", (350, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                self.display_image(result_image)
                self.status_label.config(text="✅ Detección simulada completada")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en detección: {str(e)}")
    
    def toggle_webcam(self):
        """Activa/desactiva la webcam"""
        if not self.webcam_active:
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.webcam_active = True
                    self.webcam_btn.config(text="⏹️ Detener Cámara")
                    self.status_label.config(text="📹 Cámara activa")
                    threading.Thread(target=self.webcam_loop, daemon=True).start()
                else:
                    messagebox.showerror("Error", "No se pudo acceder a la cámara")
            except Exception as e:
                messagebox.showerror("Error", f"Error iniciando cámara: {str(e)}")
        else:
            self.webcam_active = False
            if self.cap:
                self.cap.release()
            self.webcam_btn.config(text="📹 Iniciar Cámara")
            self.status_label.config(text="✅ Cámara detenida")
    
    def webcam_loop(self):
        """Loop de la webcam"""
        fps_counter = 0
        start_time = time.time()
        
        while self.webcam_active:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Opcional: ejecutar detección en tiempo real
                    if YOLO_AVAILABLE and self.model:
                        results = self.model(frame, conf=self.confidence.get())
                        frame = results[0].plot()
                    
                    self.current_image = frame
                    self.display_image(frame)
                    
                    # Calcular FPS
                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = 30 / elapsed
                        self.fps_label.config(text=f"FPS: {fps:.1f}")
                        start_time = time.time()
                        fps_counter = 0
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error en webcam: {e}")
                break
    
    def save_frame(self):
        """Guarda el frame actual"""
        if self.current_image is not None:
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".jpg",
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
                )
                if file_path:
                    cv2.imwrite(file_path, self.current_image)
                    self.status_label.config(text=f"✅ Imagen guardada: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error guardando imagen: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
    
    def show_info(self):
        """Muestra información del modelo"""
        info_text = f"""
🎯 YOLOv8 Object Detection System

👥 Desarrolladores: Herrera & Paredes
📚 Materia: Inteligencia Artificial
📋 Evaluación: Segundo Parcial

🔧 Configuración Actual:
• Confianza: {self.confidence.get():.2f}
• IoU Threshold: {self.iou_threshold.get():.2f}
• Max Detecciones: {self.max_detections.get()}

📊 Estado del Sistema:
• Modelo: {'YOLOv8 Nano' if YOLO_AVAILABLE else 'Simulación'}
• Webcam: {'Activa' if self.webcam_active else 'Inactiva'}
• Imagen Actual: {'Cargada' if self.current_image is not None else 'No cargada'}

🎓 Funcionalidades:
✅ Detección en imágenes estáticas
✅ Detección en tiempo real (webcam)
✅ Configuración de parámetros
✅ Guardado de resultados
✅ Interfaz intuitiva
        """
        
        messagebox.showinfo("Información del Sistema", info_text)
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        if self.webcam_active:
            self.webcam_active = False
            if self.cap:
                self.cap.release()
        
        messagebox.showinfo("YOLOv8 Detector", "¡Gracias por usar nuestro sistema!\\n\\nHerrera & Paredes")
        self.root.destroy()

def main():
    """Función principal"""
    print("🎯 Iniciando YOLOv8 Detector Simple...")
    print("👥 Herrera & Paredes")
    print("📚 Segundo Parcial - Inteligencia Artificial\\n")
    
    root = tk.Tk()
    app = YOLOv8SimpleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

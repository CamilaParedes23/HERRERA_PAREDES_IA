#!/usr/bin/env python3
"""
YOLOv8 Professional GUI - Herrera & Paredes
Una aplicación de escritorio profesional para detección de objetos con YOLOv8

Uso:
    python interfaz_grafica.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import os
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Colores profesionales - Esquema Material Design
COLORS = {
    'primary': '#1E88E5',        # Azul principal
    'primary_dark': '#1565C0',   # Azul oscuro
    'secondary': '#26A69A',      # Verde azulado
    'accent': '#FF7043',         # Naranja cálido
    'surface': '#FAFAFA',        # Gris muy claro
    'background': '#F5F5F5',     # Fondo principal
    'card': '#FFFFFF',           # Tarjetas blancas
    'text_primary': '#212121',   # Texto principal
    'text_secondary': '#757575', # Texto secundario
    'divider': '#E0E0E0',        # Divisores
    'success': '#4CAF50',        # Verde éxito
    'warning': '#FF9800',        # Naranja advertencia
    'error': '#F44336',          # Rojo error
    'info': '#2196F3'            # Azul información
}

class YOLOv8GUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        
        # Variables
        self.model = None
        self.current_image = None
        self.webcam_active = False
        self.cap = None
        
        # Configuración por defecto
        self.confidence = tk.DoubleVar(value=0.25)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.max_detections = tk.IntVar(value=100)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_confidence = tk.BooleanVar(value=True)
        
        # Configurar estilos
        self.setup_styles()
        
        # Configurar UI
        self.setup_ui()
        
        # Cargar modelo
        self.load_model()
        
        # Configurar eventos de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
    def setup_window(self):
        """Configura la ventana principal con estilo profesional"""
        self.root.title("YOLOv8 Professional - Object Detection Suite")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=COLORS['background'])
        
        # Icono personalizado (opcional)
        try:
            # Crear un icono simple
            icon_img = Image.new('RGBA', (32, 32), (30, 136, 229, 255))
            draw = ImageDraw.Draw(icon_img)
            draw.ellipse([8, 8, 24, 24], fill=(255, 255, 255, 255))
            draw.text((12, 10), "Y", fill=(30, 136, 229, 255))
            
            self.icon = ImageTk.PhotoImage(icon_img)
            self.root.iconphoto(True, self.icon)
        except:
            pass
    
    def setup_styles(self):
        """Configura los estilos ttk personalizados básicos"""
        try:
            self.style = ttk.Style()
            
            # Configurar tema base
            self.style.theme_use('clam')
            
            # Estilos básicos para botones
            self.style.configure(
                'Primary.TButton',
                background=COLORS['primary'],
                foreground='white',
                font=('Segoe UI', 10, 'bold')
            )
            
            self.style.configure(
                'Secondary.TButton',
                background=COLORS['secondary'],
                foreground='white',
                font=('Segoe UI', 9, 'bold')
            )
            
            self.style.configure(
                'Accent.TButton',
                background=COLORS['accent'],
                foreground='white',
                font=('Segoe UI', 9, 'bold')
            )
            
            self.style.configure(
                'Success.TButton',
                background=COLORS['success'],
                foreground='white',
                font=('Segoe UI', 9, 'bold')
            )
            
            self.style.configure(
                'Error.TButton',
                background=COLORS['error'],
                foreground='white',
                font=('Segoe UI', 9, 'bold')
            )
            
            # Estilo para scales
            self.style.configure(
                'Custom.TScale',
                background=COLORS['surface'],
                troughcolor=COLORS['divider'],
                borderwidth=0,
                lightcolor=COLORS['primary'],
                darkcolor=COLORS['primary']
            )
            
        except Exception as e:
            print(f"⚠️ Error configurando estilos: {e}")
            # Si falla, usar estilos por defecto
            self.style = ttk.Style()
        
    def setup_ui(self):
        """Configura la interfaz de usuario profesional"""
        
        # Contenedor principal con padding
        main_container = tk.Frame(self.root, bg=COLORS['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header con título elegante
        self.create_header(main_container)
        
        # Contenido principal
        content_frame = tk.Frame(main_container, bg=COLORS['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Panel lateral izquierdo - Controles
        self.create_sidebar(content_frame)
        
        # Panel central - Visualización
        self.create_main_panel(content_frame)
        
        # Barra de estado inferior
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """Crea el header profesional"""
        header_frame = tk.Frame(parent, bg=COLORS['card'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Agregar sombra visual
        shadow_frame = tk.Frame(parent, bg=COLORS['divider'], height=2)
        shadow_frame.pack(fill=tk.X, pady=(0, 18))
        
        # Contenido del header
        header_content = tk.Frame(header_frame, bg=COLORS['card'])
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Lado izquierdo - Título y subtítulo
        left_header = tk.Frame(header_content, bg=COLORS['card'])
        left_header.pack(side=tk.LEFT, fill=tk.Y)
        
        title_label = tk.Label(
            left_header,
            text="YOLOv8 Professional Suite",
            font=('Segoe UI Light', 24, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        )
        title_label.pack(anchor=tk.W)
        
        subtitle_label = tk.Label(
            left_header,
            text="Advanced Object Detection • Herrera & Paredes",
            font=('Segoe UI', 12),
            fg=COLORS['text_secondary'],
            bg=COLORS['card']
        )
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Solo título y subtítulo - Sin botones de salida
        # La aplicación se cierra usando el botón X de la ventana
    
    def create_sidebar(self, parent):
        """Crea la barra lateral con controles elegantes y scroll mejorado"""
        sidebar_frame = tk.Frame(parent, bg=COLORS['background'], width=380)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        sidebar_frame.pack_propagate(False)
        
        # Canvas y scrollbar con mejor configuración
        canvas = tk.Canvas(
            sidebar_frame, 
            bg=COLORS['background'], 
            highlightthickness=0,
            width=360
        )
        
        # Scrollbar más visible
        scrollbar = ttk.Scrollbar(
            sidebar_frame, 
            orient="vertical", 
            command=canvas.yview
        )
        
        # Frame scrollable
        scrollable_frame = tk.Frame(canvas, bg=COLORS['background'])
        
        # Configurar scroll
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        # Crear ventana del canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Configurar canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Hacer que el scrollable_frame se ajuste al ancho del canvas
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        
        # Secciones de contenido con más espaciado
        self.create_config_section(scrollable_frame)
        self.create_actions_section(scrollable_frame) 
        self.create_webcam_section(scrollable_frame)
        self.create_info_section(scrollable_frame)
        
        # Agregar espacio extra al final para mejor scroll
        tk.Frame(scrollable_frame, bg=COLORS['background'], height=50).pack(fill=tk.X)
        
        # Empaquetar elementos
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_config_section(self, parent):
        """Crea la sección de configuración elegante con mejor espaciado"""
        config_card = tk.Frame(parent, bg=COLORS['card'], relief='flat', bd=1)
        config_card.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        # Título de la sección
        title_frame = tk.Frame(config_card, bg=COLORS['card'], height=60)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="⚙️ Configuración del Modelo",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        ).pack(anchor=tk.W, pady=15)
        
        # Controles dentro de la tarjeta con más espacio
        controls_frame = tk.Frame(config_card, bg=COLORS['card'])
        controls_frame.pack(fill=tk.X, padx=20, pady=(0, 25))
        
        # Controles con mejor espaciado
        self.create_slider_control(
            controls_frame, 
            "Umbral de Confianza", 
            self.confidence, 
            0.1, 1.0, 0.05,
            "Qué tan seguro debe estar el modelo para detectar un objeto"
        )
        
        self.create_slider_control(
            controls_frame, 
            "Umbral IoU", 
            self.iou_threshold, 
            0.1, 1.0, 0.05,
            "Controla la superposición entre detecciones"
        )
        
        self.create_slider_control(
            controls_frame, 
            "Máximo de Detecciones", 
            self.max_detections, 
            10, 300, 10,
            "Número máximo de objetos a detectar"
        )
        
        # Checkboxes con mejor espaciado
        checkbox_frame = tk.Frame(controls_frame, bg=COLORS['card'])
        checkbox_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.create_elegant_checkbox(checkbox_frame, "Mostrar etiquetas de clase", self.show_labels)
        self.create_elegant_checkbox(checkbox_frame, "Mostrar nivel de confianza", self.show_confidence)
    
    def create_slider_control(self, parent, label, variable, from_, to, resolution, tooltip):
        """Crea un control deslizante elegante con mejor tamaño"""
        container = tk.Frame(parent, bg=COLORS['card'])
        container.pack(fill=tk.X, pady=(0, 20))
        
        # Etiqueta con valor actual
        label_frame = tk.Frame(container, bg=COLORS['card'])
        label_frame.pack(fill=tk.X)
        
        tk.Label(
            label_frame,
            text=label,
            font=('Segoe UI', 11, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        ).pack(side=tk.LEFT)
        
        value_label = tk.Label(
            label_frame,
            text=f"{variable.get():.2f}",
            font=('Segoe UI', 11, 'bold'),
            fg=COLORS['primary'],
            bg=COLORS['card']
        )
        value_label.pack(side=tk.RIGHT)
        
        # Slider con mejor altura
        scale = ttk.Scale(
            container,
            from_=from_, to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            length=300
        )
        scale.pack(fill=tk.X, pady=(8, 0))
        
        # Configurar resolución manualmente para ttk.Scale
        def on_scale_change(value):
            # Redondear al múltiplo más cercano de resolution
            rounded_value = round(float(value) / resolution) * resolution
            variable.set(rounded_value)
        
        scale.configure(command=on_scale_change)
        
        # Actualizar valor en tiempo real
        def update_value(*args):
            value_label.config(text=f"{variable.get():.2f}")
        
        variable.trace('w', update_value)
        
        # Tooltip
        if tooltip:
            self.create_tooltip(scale, tooltip)
    
    def create_elegant_checkbox(self, parent, text, variable):
        """Crea un checkbox elegante"""
        check_frame = tk.Frame(parent, bg=COLORS['card'])
        check_frame.pack(fill=tk.X, pady=3)
        
        checkbox = tk.Checkbutton(
            check_frame,
            text=text,
            variable=variable,
            font=('Segoe UI', 10),
            fg=COLORS['text_primary'],
            bg=COLORS['card'],
            selectcolor=COLORS['primary'],
            activebackground=COLORS['card'],
            activeforeground=COLORS['text_primary'],
            relief=tk.FLAT,
            borderwidth=0
        )
        checkbox.pack(anchor=tk.W)
    
    def create_actions_section(self, parent):
        """Crea la sección de acciones con mejor espaciado"""
        actions_card = tk.Frame(parent, bg=COLORS['card'], relief='flat', bd=1)
        actions_card.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        # Título
        title_frame = tk.Frame(actions_card, bg=COLORS['card'], height=60)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="🎯 Acciones Principales",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        ).pack(anchor=tk.W, pady=15)
        
        # Botones con mejor espaciado
        buttons_frame = tk.Frame(actions_card, bg=COLORS['card'])
        buttons_frame.pack(fill=tk.X, padx=20, pady=(0, 25))
        
        # Botones más grandes y visibles
        ttk.Button(
            buttons_frame,
            text="📁 Cargar Imagen",
            command=self.load_image
        ).pack(fill=tk.X, pady=(0, 12), ipady=8)
        
        ttk.Button(
            buttons_frame,
            text="🎨 Generar Imagen de Prueba",
            command=self.create_synthetic_image
        ).pack(fill=tk.X, pady=(0, 12), ipady=8)
        
        ttk.Button(
            buttons_frame,
            text="🚀 Ejecutar Detección",
            command=self.run_detection
        ).pack(fill=tk.X, pady=(0, 12), ipady=8)
        
        ttk.Button(
            buttons_frame,
            text="💾 Guardar Resultado",
            command=self.save_frame
        ).pack(fill=tk.X, ipady=8)
    
    def create_webcam_section(self, parent):
        """Crea la sección de webcam con mejor espaciado"""
        webcam_card = tk.Frame(parent, bg=COLORS['card'], relief='flat', bd=1)
        webcam_card.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        # Título
        title_frame = tk.Frame(webcam_card, bg=COLORS['card'], height=60)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="📹 Detección en Tiempo Real",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        ).pack(anchor=tk.W, pady=15)
        
        # Botones de webcam con mejor espaciado
        webcam_buttons_frame = tk.Frame(webcam_card, bg=COLORS['card'])
        webcam_buttons_frame.pack(fill=tk.X, padx=20, pady=(0, 25))
        
        self.webcam_btn = tk.Button(
            webcam_buttons_frame,
            text="📹 Iniciar Cámara",
            command=self.toggle_webcam,
            bg=COLORS['secondary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='raised',
            borderwidth=1,
            cursor='hand2'
        )
        self.webcam_btn.pack(fill=tk.X, pady=(0, 12), ipady=8)
        
        ttk.Button(
            webcam_buttons_frame,
            text="📸 Capturar Frame",
            command=self.save_frame
        ).pack(fill=tk.X, ipady=8)
    
    def create_info_section(self, parent):
        """Crea la sección de información con mejor espaciado"""
        info_card = tk.Frame(parent, bg=COLORS['card'], relief='flat', bd=1)
        info_card.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        # Título
        title_frame = tk.Frame(info_card, bg=COLORS['card'], height=60)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="ℹ️ Información",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        ).pack(anchor=tk.W, pady=15)
        
        # Botón de información
        info_buttons_frame = tk.Frame(info_card, bg=COLORS['card'])
        info_buttons_frame.pack(fill=tk.X, padx=20, pady=(0, 25))
        
        ttk.Button(
            info_buttons_frame,
            text="📊 Detalles del Modelo",
            command=self.show_model_info
        ).pack(fill=tk.X, ipady=8)
    
    def create_main_panel(self, parent):
        """Crea el panel principal de visualización"""
        main_panel = tk.Frame(parent, bg=COLORS['card'], relief='flat', bd=1)
        main_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Header del panel principal
        panel_header = tk.Frame(main_panel, bg=COLORS['card'], height=60)
        panel_header.pack(fill=tk.X, padx=20, pady=(20, 0))
        panel_header.pack_propagate(False)
        
        tk.Label(
            panel_header,
            text="🖼️ Área de Visualización",
            font=('Segoe UI', 16, 'bold'),
            fg=COLORS['text_primary'],
            bg=COLORS['card']
        ).pack(anchor=tk.W, pady=15)
        
        # Canvas container con padding elegante
        canvas_container = tk.Frame(main_panel, bg=COLORS['surface'], relief='flat', bd=1)
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        # Canvas para la imagen
        self.canvas = tk.Canvas(
            canvas_container,
            bg=COLORS['surface'],
            highlightthickness=0,
            relief='flat'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mensaje inicial elegante
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Muestra un mensaje de bienvenida elegante"""
        self.canvas.delete("all")
        
        # Obtener dimensiones del canvas
        self.canvas.update()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width, height = 800, 600
        
        # Crear mensaje centrado
        center_x, center_y = width // 2, height // 2
        
        # Icono principal
        self.canvas.create_text(
            center_x, center_y - 60,
            text="🎯",
            font=('Segoe UI', 48),
            fill=COLORS['primary']
        )
        
        # Título
        self.canvas.create_text(
            center_x, center_y - 10,
            text="YOLOv8 Object Detection",
            font=('Segoe UI', 20, 'bold'),
            fill=COLORS['text_primary']
        )
        
        # Subtítulo
        self.canvas.create_text(
            center_x, center_y + 20,
            text="Carga una imagen o inicia la cámara para comenzar",
            font=('Segoe UI', 12),
            fill=COLORS['text_secondary']
        )
        
        # Instrucciones
        instructions = [
            "📁 Usa 'Cargar Imagen' para analizar una foto",
            "🎨 Prueba 'Generar Imagen de Prueba' para una demo",
            "� Inicia 'Cámara' para detección en tiempo real",
            "⚙️ Ajusta la configuración según tus necesidades"
        ]
        
        for i, instruction in enumerate(instructions):
            self.canvas.create_text(
                center_x, center_y + 60 + (i * 25),
                text=instruction,
                font=('Segoe UI', 10),
                fill=COLORS['text_secondary']
            )
    
    def create_status_bar(self, parent):
        """Crea la barra de estado profesional"""
        status_card = tk.Frame(parent, bg=COLORS['card'], height=50, relief='flat', bd=1)
        status_card.pack(fill=tk.X, pady=(20, 0))
        status_card.pack_propagate(False)
        
        status_content = tk.Frame(status_card, bg=COLORS['card'])
        status_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Estado del modelo (izquierda)
        self.status_label = tk.Label(
            status_content,
            text="✅ Modelo YOLOv8 listo para usar",
            font=('Segoe UI', 10),
            fg=COLORS['success'],
            bg=COLORS['card']
        )
        self.status_label.pack(side=tk.LEFT)
        
        # FPS y rendimiento (centro)
        self.performance_label = tk.Label(
            status_content,
            text="",
            font=('Segoe UI', 10),
            fg=COLORS['info'],
            bg=COLORS['card']
        )
        self.performance_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Información adicional (derecha)
        self.fps_label = tk.Label(
            status_content,
            text="Listo para detectar",
            font=('Segoe UI', 10),
            fg=COLORS['text_secondary'],
            bg=COLORS['card']
        )
        self.fps_label.pack(side=tk.RIGHT)
    
    def create_tooltip(self, widget, text):
        """Crea un tooltip elegante para un widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            tooltip.configure(bg=COLORS['text_primary'])
            
            label = tk.Label(
                tooltip,
                text=text,
                bg=COLORS['text_primary'],
                fg='white',
                font=('Segoe UI', 9),
                padx=10,
                pady=5
            )
            label.pack()
            
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        
    def load_model(self):
        """Carga el modelo YOLOv8"""
        try:
            if not YOLO_AVAILABLE:
                messagebox.showerror("Error", "Ultralytics no está instalado.\nEjecuta: pip install ultralytics")
                return
                
            self.update_status("🔄 Cargando modelo YOLOv8...")
            self.model = YOLO('yolov8n.pt')
            self.update_status("✅ Modelo YOLOv8 cargado exitosamente")
            
        except Exception as e:
            self.update_status(f"❌ Error cargando modelo: {str(e)}")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")
    
    def update_status(self, message):
        """Actualiza el mensaje de estado"""
        self.status_label.config(text=message)
        self.root.update()
    
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
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
                    return
                
                self.display_image(self.current_image, "Imagen Cargada")
                self.update_status(f"📁 Imagen cargada: {Path(file_path).name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen:\n{str(e)}")
    
    def create_synthetic_image(self):
        """Crea una imagen sintética para pruebas"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Simular objetos detectables
        # Persona (rectángulo vertical + cabeza)
        cv2.rectangle(img, (50, 150), (150, 400), (100, 150, 200), -1)
        cv2.circle(img, (100, 120), 25, (200, 180, 150), -1)
        
        # Carro
        cv2.rectangle(img, (300, 250), (500, 350), (50, 50, 150), -1)
        cv2.circle(img, (330, 350), 20, (0, 0, 0), -1)  # rueda
        cv2.circle(img, (470, 350), 20, (0, 0, 0), -1)  # rueda
        
        # Animal
        cv2.ellipse(img, (200, 350), (60, 30), 0, 0, 360, (139, 69, 19), -1)
        
        # Texto
        cv2.putText(img, "YOLO DEMO", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        self.current_image = img
        self.display_image(img, "Imagen Sintética")
        self.update_status("🎨 Imagen sintética creada")
    
    def run_detection(self):
        """Ejecuta detección en la imagen actual"""
        if self.current_image is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen o crea una imagen sintética")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        try:
            self.update_status("🔍 Ejecutando detección...")
            
            start_time = time.time()
            results = self.model.predict(
                self.current_image,
                conf=self.confidence.get(),
                iou=self.iou_threshold.get(),
                max_det=self.max_detections.get(),
                verbose=False
            )
            end_time = time.time()
            
            inference_time = end_time - start_time
            fps = 1 / inference_time
            
            # Procesar resultados
            annotated_img = results[0].plot()
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            # Mostrar imagen con detecciones
            self.display_image(annotated_img, f"Detecciones: {detections}")
            
            # Actualizar estado
            self.update_status(f"✅ Detección completada: {detections} objetos en {inference_time:.3f}s")
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            # Mostrar detalles
            if detections > 0:
                details = []
                for i, box in enumerate(results[0].boxes):
                    if box.cls is not None:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        details.append(f"{i+1}. {class_name}: {confidence:.2f}")
                
                messagebox.showinfo("Detecciones", f"Objetos encontrados:\n\n" + "\n".join(details))
            
        except Exception as e:
            self.update_status(f"❌ Error en detección: {str(e)}")
            messagebox.showerror("Error", f"Error durante detección:\n{str(e)}")
    
    def display_image(self, img, title=""):
        """Muestra una imagen en el canvas"""
        # Redimensionar imagen para que quepa en el canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 600, 400
        
        img_height, img_width = img.shape[:2]
        
        # Calcular factor de escala
        scale_x = (canvas_width - 20) / img_width
        scale_y = (canvas_height - 20) / img_height
        scale = min(scale_x, scale_y)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Redimensionar
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Convertir BGR a RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Convertir a PIL Image y luego a PhotoImage
        pil_img = Image.fromarray(rgb_img)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # Limpiar canvas y mostrar imagen
        self.canvas.delete("all")
        
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        if title:
            self.canvas.create_text(
                canvas_width // 2, 10,
                text=title,
                font=('Arial', 12, 'bold'),
                fill='#ECF0F1'
            )
    
    def toggle_webcam(self):
        """Inicia o detiene la webcam"""
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Inicia la webcam"""
        try:
            print("📹 Intentando iniciar webcam...")
            
            # Intentar diferentes índices de cámara
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    # Probar si realmente puede leer frames
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"✅ Webcam encontrada en índice {camera_index}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    self.cap = None
            
            if not self.cap or not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir ninguna webcam.\n\nVerifica que:\n• Tengas una cámara conectada\n• No esté siendo usada por otra aplicación\n• Los drivers estén instalados")
                return
            
            # Configurar cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.webcam_active = True
            self.webcam_btn.config(
                text="⏹️ Detener Webcam",
                bg=COLORS['error']
            )
            self.update_status("📹 Webcam iniciada exitosamente")
            
            # Iniciar hilo de webcam
            self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.webcam_thread.start()
            
        except Exception as e:
            print(f"❌ Error iniciando webcam: {e}")
            messagebox.showerror("Error", f"Error iniciando webcam:\n{str(e)}\n\nVerifica que la cámara no esté siendo usada por otra aplicación.")
    
    def stop_webcam(self):
        """Detiene la webcam"""
        self.webcam_active = False
        if self.cap:
            self.cap.release()
        
        # Configurar botón correctamente
        self.webcam_btn.config(
            text="📹 Iniciar Cámara",
            bg=COLORS['secondary']
        )
        self.update_status("⏹️ Webcam detenida")
    
    def webcam_loop(self):
        """Bucle principal de la webcam"""
        print("🎥 Iniciando bucle de webcam...")
        frame_count = 0
        
        while self.webcam_active:
            try:
                if not self.cap or not self.cap.isOpened():
                    print("❌ Cámara no disponible")
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ No se pudo leer frame de la cámara")
                    break
                
                frame_count += 1
                
                # Ejecutar detección si el modelo está disponible
                if self.model:
                    start_time = time.time()
                    try:
                        results = self.model.predict(
                            frame,
                            conf=self.confidence.get(),
                            verbose=False
                        )
                        end_time = time.time()
                        
                        annotated_frame = results[0].plot()
                        detections = len(results[0].boxes) if results[0].boxes is not None else 0
                        
                        inference_time = end_time - start_time
                        fps = 1 / inference_time if inference_time > 0 else 0
                        
                        # Actualizar UI en el hilo principal
                        self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f} | Frame: {frame_count}"))
                        
                        self.current_image = annotated_frame
                        self.root.after(0, lambda: self.display_image(annotated_frame, f"Webcam - {detections} objetos"))
                        
                    except Exception as e:
                        print(f"⚠️ Error en detección: {e}")
                        self.current_image = frame
                        self.root.after(0, lambda: self.display_image(frame, "Webcam (sin detección)"))
                else:
                    self.current_image = frame
                    self.root.after(0, lambda: self.display_image(frame, f"Webcam - Frame {frame_count}"))
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error en bucle de webcam: {e}")
                break
        
        print("🛑 Bucle de webcam terminado")
        self.root.after(0, self.stop_webcam)
    
    def save_frame(self):
        """Guarda el frame actual"""
        if self.current_image is None:
            messagebox.showwarning("Aviso", "No hay imagen para guardar")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Guardar imagen",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_image)
                self.update_status(f"💾 Imagen guardada: {Path(file_path).name}")
                messagebox.showinfo("Éxito", f"Imagen guardada en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error guardando imagen:\n{str(e)}")
    
    def show_model_info(self):
        """Muestra información del modelo"""
        if not self.model:
            messagebox.showerror("Error", "Modelo no cargado")
            return
        
        info = f"""🤖 INFORMACIÓN DEL MODELO YOLOV8
        
📱 Tipo: YOLOv8 Nano
🎯 Clases disponibles: {len(self.model.names)}
🏷️ Dataset: COCO (Common Objects in Context)
⚙️ Device: {'GPU' if hasattr(self.model, 'device') else 'CPU'}

🔧 CONFIGURACIÓN ACTUAL:
• Confianza: {self.confidence.get():.2f}
• IoU Threshold: {self.iou_threshold.get():.2f}
• Max Detecciones: {self.max_detections.get()}

🏷️ ALGUNAS CLASES DETECTABLES:
person, bicycle, car, motorcycle, airplane,
bus, train, truck, boat, traffic light,
fire hydrant, stop sign, parking meter,
bench, bird, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee...

📊 CAPACIDADES:
✅ Detección en tiempo real
✅ 80 clases diferentes
✅ Ajuste de parámetros dinámico
✅ Múltiples fuentes de entrada
✅ Exportación de resultados"""
        
        messagebox.showinfo("Información del Modelo", info)
    
    def on_window_close(self):
        """Maneja el cierre desde el botón X de la ventana"""
        print("❌ Cerrando aplicación...")
        try:
            # Detener webcam si está activa
            if hasattr(self, 'webcam_active') and self.webcam_active:
                self.webcam_active = False
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            
            # Cerrar aplicación directamente
            self.root.destroy()
        except:
            import sys
            sys.exit(0)

def main():
    """Función principal"""
    print("🎯 Iniciando Interfaz Gráfica YOLOv8...")
    print("👥 Proyecto: Herrera & Paredes")
    print("📚 Segundo Parcial - Inteligencia Artificial")
    print()
    
    root = tk.Tk()
    app = YOLOv8GUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Cerrando aplicación...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

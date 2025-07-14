"""
Script de Detección en Tiempo Real para YOLOv8
Estudiantes: Herrera & Paredes

Este script demuestra el funcionamiento del modelo YOLOv8 entrenado
realizando detección de objetos en imágenes, videos y cámara web.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from pathlib import Path
import time
from datetime import datetime
import argparse

class YOLODetector:
    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.45):
        """
        Inicializa el detector YOLOv8
        
        Args:
            model_path (str): Ruta al modelo entrenado
            confidence_threshold (float): Umbral de confianza para detecciones
            iou_threshold (float): Umbral IoU para NMS
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = []
        self.colors = None
        
        # Crear directorio para resultados
        self.results_dir = Path('results/detections')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Inicializando YOLODetector")
        print(f"🤖 Modelo: {model_path}")
        print(f"🎚️ Confianza: {confidence_threshold}")
        print(f"🔄 IoU: {iou_threshold}")
    
    def load_model(self):
        """Carga el modelo entrenado"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ No se encontró el modelo en: {self.model_path}")
                return False
            
            self.model = YOLO(self.model_path)
            print(f"✅ Modelo cargado exitosamente")
            
            # Obtener nombres de clases
            self.class_names = list(self.model.names.values())
            print(f"🏷️ Clases disponibles: {self.class_names}")
            
            # Generar colores únicos para cada clase
            self.colors = self.generate_colors(len(self.class_names))
            
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return False
    
    def generate_colors(self, num_classes):
        """Genera colores únicos para cada clase"""
        colors = []
        for i in range(num_classes):
            # Generar colores en HSV y convertir a RGB
            hue = i / num_classes
            color = plt.cm.hsv(hue)
            colors.append([int(c * 255) for c in color[:3]])
        return colors
    
    def detect_image(self, image_path, save_result=True, show_result=True):
        """
        Detecta objetos en una imagen
        
        Args:
            image_path (str): Ruta a la imagen
            save_result (bool): Guardar resultado
            show_result (bool): Mostrar resultado
        """
        if self.model is None:
            print("❌ Modelo no cargado")
            return None
        
        try:
            print(f"🖼️ Procesando imagen: {image_path}")
            
            # Ejecutar detección
            results = self.model.predict(
                image_path,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Procesar resultados
            result = results[0]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Dibujar detecciones
            annotated_image = self.draw_detections(image_rgb, result)
            
            # Mostrar estadísticas
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"✅ Detecciones encontradas: {num_detections}")
            
            if num_detections > 0:
                self.print_detection_summary(result)
            
            # Guardar resultado
            if save_result:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = Path(image_path).stem
                save_path = self.results_dir / f"detection_{filename}_{timestamp}.jpg"
                
                # Convertir de RGB a BGR para OpenCV
                annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), annotated_bgr)
                print(f"💾 Resultado guardado en: {save_path}")
            
            # Mostrar resultado
            if show_result:
                self.show_image_result(annotated_image, f"Detección: {Path(image_path).name}")
            
            return annotated_image, result
            
        except Exception as e:
            print(f"❌ Error al procesar imagen: {e}")
            return None
    
    def draw_detections(self, image, result):
        """Dibuja las detecciones en la imagen"""
        annotated_image = image.copy()
        
        if result.boxes is None or len(result.boxes) == 0:
            return annotated_image
        
        # Convertir a numpy array para OpenCV
        annotated_image = np.array(annotated_image)
        
        # Dibujar cada detección
        for box in result.boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Obtener información de la clase
            class_name = self.class_names[class_id]
            color = self.colors[class_id]
            
            # Dibujar rectángulo
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Preparar texto
            label = f"{class_name}: {confidence:.2f}"
            
            # Calcular tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Dibujar fondo del texto
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - baseline - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_image
    
    def print_detection_summary(self, result):
        """Imprime resumen de detecciones"""
        if result.boxes is None:
            return
        
        # Contar detecciones por clase
        class_counts = {}
        confidences = []
        
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()
            class_name = self.class_names[class_id]
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)
        
        print("\n📊 RESUMEN DE DETECCIONES:")
        for class_name, count in class_counts.items():
            print(f"   • {class_name}: {count} objeto(s)")
        
        print(f"🎯 Confianza promedio: {np.mean(confidences):.3f}")
        print(f"🔝 Confianza máxima: {max(confidences):.3f}")
        print(f"🔻 Confianza mínima: {min(confidences):.3f}")
    
    def show_image_result(self, image, title="Detección"):
        """Muestra el resultado en una ventana"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def detect_video(self, video_path, save_result=True, show_real_time=True):
        """
        Detecta objetos en un video
        
        Args:
            video_path (str): Ruta al video
            save_result (bool): Guardar video resultado
            show_real_time (bool): Mostrar en tiempo real
        """
        if self.model is None:
            print("❌ Modelo no cargado")
            return
        
        try:
            print(f"🎬 Procesando video: {video_path}")
            
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ No se pudo abrir el video: {video_path}")
                return
            
            # Obtener propiedades del video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Configurar writer para guardar resultado
            if save_result:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = Path(video_path).stem
                output_path = self.results_dir / f"detection_{filename}_{timestamp}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Ejecutar detección
                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Anotar frame
                annotated_frame = results[0].plot()
                
                # Guardar frame si es necesario
                if save_result:
                    out.write(annotated_frame)
                
                # Mostrar en tiempo real
                if show_real_time:
                    cv2.imshow('YOLOv8 Detection', annotated_frame)
                    
                    # Presionar 'q' para salir
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Mostrar progreso
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    fps_actual = frame_count / elapsed_time
                    print(f"⏳ Progreso: {progress:.1f}% - FPS: {fps_actual:.1f}")
            
            # Limpiar recursos
            cap.release()
            if save_result:
                out.release()
                print(f"💾 Video guardado en: {output_path}")
            
            if show_real_time:
                cv2.destroyAllWindows()
            
            # Estadísticas finales
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"✅ Video procesado en {total_time:.2f} segundos")
            print(f"📊 FPS promedio: {avg_fps:.2f}")
            
        except Exception as e:
            print(f"❌ Error al procesar video: {e}")
    
    def detect_webcam(self, camera_index=0):
        """
        Detecta objetos usando la cámara web
        
        Args:
            camera_index (int): Índice de la cámara
        """
        if self.model is None:
            print("❌ Modelo no cargado")
            return
        
        try:
            print(f"📷 Iniciando detección con cámara web (índice: {camera_index})")
            print("💡 Presiona 'q' para salir, 's' para guardar frame actual")
            
            # Abrir cámara
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"❌ No se pudo abrir la cámara con índice: {camera_index}")
                return
            
            # Configurar resolución (opcional)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ No se pudo leer frame de la cámara")
                    break
                
                frame_count += 1
                
                # Ejecutar detección
                results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Anotar frame
                annotated_frame = results[0].plot()
                
                # Añadir información FPS
                if frame_count > 1:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Guardar frame actual
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = self.results_dir / f"webcam_capture_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), annotated_frame)
                    print(f"📸 Frame guardado en: {save_path}")
            
            # Limpiar recursos
            cap.release()
            cv2.destroyAllWindows()
            
            # Estadísticas finales
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"✅ Sesión de cámara finalizada")
            print(f"📊 FPS promedio: {avg_fps:.2f}")
            print(f"⏱️ Tiempo total: {total_time:.2f} segundos")
            
        except Exception as e:
            print(f"❌ Error con la cámara web: {e}")

def main():
    """Función principal con interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(description="Detector YOLOv8 - Herrera & Paredes")
    parser.add_argument("--model", default="models/yolo_custom_final.pt", 
                       help="Ruta al modelo entrenado")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Umbral de confianza (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="Umbral IoU para NMS (default: 0.45)")
    parser.add_argument("--image", type=str, help="Ruta a imagen para detectar")
    parser.add_argument("--video", type=str, help="Ruta a video para detectar")
    parser.add_argument("--webcam", action="store_true", help="Usar cámara web")
    parser.add_argument("--camera_index", type=int, default=0, 
                       help="Índice de cámara (default: 0)")
    
    args = parser.parse_args()
    
    print("🎯 DETECTOR DE OBJETOS YOLOv8 - HERRERA & PAREDES")
    print("=" * 60)
    
    # Inicializar detector
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    # Cargar modelo
    if not detector.load_model():
        print("❌ No se pudo cargar el modelo")
        return
    
    # Ejecutar según argumentos
    if args.image:
        if os.path.exists(args.image):
            detector.detect_image(args.image)
        else:
            print(f"❌ No se encontró la imagen: {args.image}")
    
    elif args.video:
        if os.path.exists(args.video):
            detector.detect_video(args.video)
        else:
            print(f"❌ No se encontró el video: {args.video}")
    
    elif args.webcam:
        detector.detect_webcam(args.camera_index)
    
    else:
        print("💡 Uso interactivo:")
        print("1. Imagen: python detect.py --image ruta/imagen.jpg")
        print("2. Video: python detect.py --video ruta/video.mp4")
        print("3. Webcam: python detect.py --webcam")
        
        # Modo demo si no se especifican argumentos
        print("\n🎪 Iniciando modo demo con webcam...")
        detector.detect_webcam()

if __name__ == "__main__":
    main()

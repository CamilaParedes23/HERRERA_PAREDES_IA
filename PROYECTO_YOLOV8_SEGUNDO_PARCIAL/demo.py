"""
Demo Rápido - Proyecto YOLOv8
Estudiantes: Herrera & Paredes

Este script proporciona una demostración rápida del proyecto
sin necesidad de entrenar el modelo.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        print("✅ Dependencias verificadas correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error de dependencias: {e}")
        print("💡 Ejecuta 'pip install -r requirements.txt' primero")
        return False

# Importar YOLO globalmente después de verificar dependencias
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def demo_webcam():
    """Demostración con webcam usando modelo preentrenado"""
    print("🚀 DEMO RÁPIDO CON WEBCAM")
    print("=" * 40)
    print("💡 Presiona 'q' para salir")
    print("💡 Presiona 's' para guardar frame")
    print()
    
    if YOLO is None:
        print("❌ YOLO no está disponible. Instala ultralytics primero.")
        return
    
    try:
        # Cargar modelo preentrenado
        print("📥 Cargando modelo YOLOv8...")
        model = YOLO('yolov8n.pt')  # Descarga automáticamente si no existe
        print("✅ Modelo cargado")
        
        # Abrir webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return
        
        print("📷 Cámara iniciada - ¡Muestra objetos para detectar!")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Ejecutar detección
            results = model.predict(frame, conf=0.3, verbose=False)
            
            # Anotar frame
            annotated_frame = results[0].plot()
            
            # Calcular FPS
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                
                # Añadir información FPS
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Añadir instrucciones
            cv2.putText(annotated_frame, "Presiona 'q' para salir", 
                      (10, annotated_frame.shape[0] - 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Presiona 's' para guardar", 
                      (10, annotated_frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow('YOLOv8 Demo - Herrera & Paredes', annotated_frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Guardar frame
                save_path = f"demo_capture_{int(time.time())}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"📸 Frame guardado: {save_path}")
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\n📊 Estadísticas de la sesión:")
        print(f"   • Frames procesados: {frame_count}")
        print(f"   • Tiempo total: {total_time:.2f} segundos")
        print(f"   • FPS promedio: {avg_fps:.2f}")
        
    except Exception as e:
        print(f"❌ Error durante demo: {e}")

def demo_image():
    """Demostración con imagen de ejemplo"""
    print("🖼️ DEMO CON IMAGEN DE EJEMPLO")
    print("=" * 40)
    
    if YOLO is None:
        print("❌ YOLO no está disponible. Instala ultralytics primero.")
        return
    
    try:
        # Cargar modelo
        print("📥 Cargando modelo YOLOv8...")
        model = YOLO('yolov8n.pt')
        
        # Crear imagen de ejemplo
        print("🎨 Creando imagen de ejemplo...")
        
        # Imagen sintética con formas geométricas
        img = np.ones((640, 640, 3), dtype=np.uint8) * 200
        
        # Añadir formas que podrían ser detectadas como objetos
        cv2.rectangle(img, (100, 200), (250, 350), (0, 0, 255), -1)  # Rectángulo rojo
        cv2.circle(img, (450, 150), 80, (0, 255, 0), -1)  # Círculo verde
        cv2.rectangle(img, (300, 400), (500, 550), (255, 0, 0), -1)  # Rectángulo azul
        
        # Añadir texto
        cv2.putText(img, "DEMO IMAGE", (200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Ejecutar detección
        print("🔍 Ejecutando detección...")
        results = model.predict(img, conf=0.25, verbose=False)
        
        # Obtener imagen con anotaciones
        annotated_img = results[0].plot()
        
        # Mostrar imagen original y resultado
        combined = np.hstack([img, annotated_img])
        
        # Redimensionar para mejor visualización
        height, width = combined.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            combined = cv2.resize(combined, (new_width, new_height))
        
        cv2.imshow('Demo: Original vs Detectado - Herrera & Paredes', combined)
        
        print("💡 Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Guardar resultado
        save_path = f"demo_result_{int(time.time())}.jpg"
        cv2.imwrite(save_path, combined)
        print(f"💾 Resultado guardado: {save_path}")
        
    except Exception as e:
        print(f"❌ Error durante demo: {e}")

def show_model_info():
    """Muestra información del modelo"""
    print("🤖 INFORMACIÓN DEL MODELO")
    print("=" * 40)
    
    if YOLO is None:
        print("❌ YOLO no está disponible. Instala ultralytics primero.")
        return
    
    try:
        model = YOLO('yolov8n.pt')
        
        print(f"📋 Modelo: YOLOv8 Nano")
        print(f"🏷️ Clases disponibles: {len(model.names)}")
        print(f"📊 Primeras 10 clases:")
        
        for i in range(min(10, len(model.names))):
            print(f"   {i}: {model.names[i]}")
        
        if len(model.names) > 10:
            print(f"   ... y {len(model.names) - 10} clases más")
        
        print(f"\n💾 Tamaño del modelo: ~6 MB")
        print(f"⚡ Velocidad: ~60 FPS (GPU), ~25 FPS (CPU)")
        
    except Exception as e:
        print(f"❌ Error obteniendo info: {e}")

def main():
    """Función principal del demo"""
    print("🎯 DEMO RÁPIDO - PROYECTO YOLOV8")
    print("Estudiantes: Herrera & Paredes")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_dependencies():
        return
    
    while True:
        print("\n🎮 OPCIONES DE DEMO:")
        print("1. 📷 Demo con webcam (recomendado)")
        print("2. 🖼️ Demo con imagen sintética")
        print("3. 🤖 Información del modelo")
        print("4. 🚪 Salir")
        
        try:
            choice = input("\n👉 Elige una opción (1-4): ").strip()
            
            if choice == '1':
                demo_webcam()
            elif choice == '2':
                demo_image()
            elif choice == '3':
                show_model_info()
            elif choice == '4':
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción inválida. Elige 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

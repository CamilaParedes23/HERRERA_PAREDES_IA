"""
Script de Entrenamiento para YOLOv8
Estudiantes: Herrera & Paredes

Este script entrena un modelo YOLOv8 personalizado para detección de objetos.
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class YOLOTrainer:
    def __init__(self, model_size='yolov8n', data_config='data/dataset.yaml'):
        """
        Inicializa el entrenador YOLOv8
        
        Args:
            model_size (str): Tamaño del modelo ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            data_config (str): Ruta al archivo de configuración del dataset
        """
        self.model_size = model_size
        self.data_config = data_config
        self.model = None
        self.results = None
        
        # Crear directorios necesarios
        self.models_dir = Path('models')
        self.results_dir = Path('results')
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Inicializando YOLOTrainer con modelo {model_size}")
        print(f"📊 Configuración de datos: {data_config}")
        print(f"🔧 Device disponible: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    def setup_model(self):
        """Configura el modelo YOLOv8"""
        try:
            # Cargar modelo preentrenado
            self.model = YOLO(f'{self.model_size}.pt')
            print(f"✅ Modelo {self.model_size} cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return False
    
    def train(self, epochs=100, imgsz=640, batch=16, patience=20, save_period=10):
        """
        Entrena el modelo YOLOv8
        
        Args:
            epochs (int): Número de épocas de entrenamiento
            imgsz (int): Tamaño de imagen para entrenamiento
            batch (int): Tamaño del batch
            patience (int): Paciencia para early stopping
            save_period (int): Cada cuántas épocas guardar el modelo
        """
        if self.model is None:
            print("❌ Modelo no inicializado. Ejecuta setup_model() primero.")
            return False
        
        print(f"🏋️ Iniciando entrenamiento...")
        print(f"📈 Épocas: {epochs}")
        print(f"🖼️ Tamaño de imagen: {imgsz}")
        print(f"📦 Batch size: {batch}")
        
        try:
            # Configurar parámetros de entrenamiento
            self.results = self.model.train(
                data=self.data_config,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                save_period=save_period,
                device='0' if torch.cuda.is_available() else 'cpu',
                project='results',
                name=f'yolo_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                exist_ok=True,
                verbose=True,
                plots=True
            )
            
            print("✅ Entrenamiento completado exitosamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            return False
    
    def save_model(self, custom_name=None):
        """Guarda el modelo entrenado"""
        if self.model is None:
            print("❌ No hay modelo para guardar")
            return False
        
        try:
            if custom_name:
                model_path = self.models_dir / f"{custom_name}.pt"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = self.models_dir / f"yolo_custom_{timestamp}.pt"
            
            # Exportar modelo
            self.model.export(format='onnx')  # También exportar a ONNX
            
            print(f"💾 Modelo guardado en: {model_path}")
            return str(model_path)
            
        except Exception as e:
            print(f"❌ Error al guardar el modelo: {e}")
            return False
    
    def get_training_metrics(self):
        """Obtiene las métricas de entrenamiento"""
        if self.results is None:
            print("❌ No hay resultados de entrenamiento disponibles")
            return None
        
        # Extraer métricas principales
        metrics = {
            'mAP50': self.results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': self.results.results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': self.results.results_dict.get('metrics/precision(B)', 0),
            'recall': self.results.results_dict.get('metrics/recall(B)', 0),
            'train_loss': self.results.results_dict.get('train/box_loss', 0),
            'val_loss': self.results.results_dict.get('val/box_loss', 0)
        }
        
        return metrics
    
    def create_training_summary(self):
        """Crea un resumen del entrenamiento"""
        metrics = self.get_training_metrics()
        if metrics is None:
            return
        
        # Crear reporte de métricas
        report = f"""
    🏆 RESUMEN DE ENTRENAMIENTO - YOLOv8
    ===================================
    
    📊 MÉTRICAS FINALES:
    • mAP@0.5: {metrics['mAP50']:.4f}
    • mAP@0.5:0.95: {metrics['mAP50-95']:.4f}
    • Precisión: {metrics['precision']:.4f}
    • Recall: {metrics['recall']:.4f}
    • Train Loss: {metrics['train_loss']:.4f}
    • Validation Loss: {metrics['val_loss']:.4f}
    
    🔧 CONFIGURACIÓN:
    • Modelo: {self.model_size}
    • Dataset: {self.data_config}
    • Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}
    
    📅 Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        print(report)
        
        # Guardar reporte
        report_path = self.results_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 Reporte guardado en: {report_path}")

def main():
    """Función principal para ejecutar el entrenamiento"""
    print("🎯 PROYECTO DE DETECCIÓN DE OBJETOS - HERRERA & PAREDES")
    print("=" * 60)
    
    # Verificar si existe la configuración del dataset
    if not os.path.exists('data/dataset.yaml'):
        print("⚠️ No se encontró data/dataset.yaml")
        print("📝 Creando configuración de ejemplo...")
        create_sample_dataset_config()
    
    # Inicializar entrenador
    trainer = YOLOTrainer(model_size='yolov8n')  # Usar modelo nano para pruebas rápidas
    
    # Configurar modelo
    if trainer.setup_model():
        print("🎯 Iniciando proceso de entrenamiento...")
        
        # Entrenar (usar parámetros más pequeños para pruebas)
        if trainer.train(epochs=50, batch=8, imgsz=640):
            # Guardar modelo
            trainer.save_model("yolo_custom_final")
            
            # Crear resumen
            trainer.create_training_summary()
            
            print("🎉 ¡Entrenamiento completado exitosamente!")
        else:
            print("❌ Error durante el entrenamiento")
    else:
        print("❌ Error al configurar el modelo")

def create_sample_dataset_config():
    """Crea una configuración de dataset de ejemplo"""
    config = {
        'path': '../data',  # Ruta relativa al dataset
        'train': 'images/train',  # Carpeta de imágenes de entrenamiento
        'val': 'images/val',      # Carpeta de imágenes de validación
        'test': 'images/test',    # Carpeta de imágenes de prueba (opcional)
        
        # Nombres de las clases (personalizar según tu dataset)
        'names': {
            0: 'person',
            1: 'car',
            2: 'truck',
            3: 'bus',
            4: 'motorcycle',
            5: 'bicycle'
        }
    }
    
    # Crear directorio data si no existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar configuración
    with open('data/dataset.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Archivo dataset.yaml creado")
    print("📝 Modifica las clases según tu dataset personalizado")

if __name__ == "__main__":
    main()

"""
Script de Evaluación para YOLOv8
Estudiantes: Herrera & Paredes

Este script evalúa el rendimiento del modelo YOLOv8 entrenado
y genera métricas detalladas de evaluación.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

class YOLOEvaluator:
    def __init__(self, model_path, data_config='data/dataset.yaml'):
        """
        Inicializa el evaluador YOLOv8
        
        Args:
            model_path (str): Ruta al modelo entrenado
            data_config (str): Ruta al archivo de configuración del dataset
        """
        self.model_path = model_path
        self.data_config = data_config
        self.model = None
        self.results = None
        self.class_names = []
        
        # Crear directorio para resultados
        self.results_dir = Path('results/evaluation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Inicializando YOLOEvaluator")
        print(f"🤖 Modelo: {model_path}")
        print(f"📁 Datos: {data_config}")
    
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
            print(f"🏷️ Clases detectadas: {self.class_names}")
            
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            return False
    
    def validate_model(self):
        """Ejecuta validación del modelo y obtiene métricas"""
        if self.model is None:
            print("❌ Modelo no cargado")
            return False
        
        try:
            print("🔍 Ejecutando validación del modelo...")
            
            # Ejecutar validación
            self.results = self.model.val(
                data=self.data_config,
                plots=True,
                save_json=True,
                project=str(self.results_dir),
                name='validation'
            )
            
            print("✅ Validación completada")
            return True
            
        except Exception as e:
            print(f"❌ Error durante la validación: {e}")
            return False
    
    def get_detailed_metrics(self):
        """Obtiene métricas detalladas del modelo"""
        if self.results is None:
            print("❌ No hay resultados de validación")
            return None
        
        # Extraer métricas principales
        metrics = {
            # Métricas globales
            'mAP50': float(self.results.box.map50),
            'mAP50_95': float(self.results.box.map),
            'precision': float(self.results.box.mp),
            'recall': float(self.results.box.mr),
            
            # Métricas por clase
            'precision_per_class': self.results.box.p.tolist() if hasattr(self.results.box, 'p') else [],
            'recall_per_class': self.results.box.r.tolist() if hasattr(self.results.box, 'r') else [],
            'map50_per_class': self.results.box.ap50.tolist() if hasattr(self.results.box, 'ap50') else [],
            'map50_95_per_class': self.results.box.ap.tolist() if hasattr(self.results.box, 'ap') else [],
            
            # Información adicional
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_images': getattr(self.results, 'seen', 0),
            'total_instances': getattr(self.results, 'nt_per_class', {}).get('total', 0) if hasattr(self.results, 'nt_per_class') else 0
        }
        
        return metrics
    
    def create_metrics_report(self):
        """Crea un reporte detallado de métricas"""
        metrics = self.get_detailed_metrics()
        if metrics is None:
            return
        
        # Crear DataFrame para métricas por clase
        if metrics['precision_per_class']:
            class_metrics_df = pd.DataFrame({
                'Clase': metrics['class_names'],
                'Precisión': metrics['precision_per_class'],
                'Recall': metrics['recall_per_class'],
                'mAP@0.5': metrics['map50_per_class'],
                'mAP@0.5:0.95': metrics['map50_95_per_class']
            })
        else:
            class_metrics_df = pd.DataFrame()
        
        # Crear reporte de texto
        report = f"""
🏆 REPORTE DE EVALUACIÓN - YOLOv8
================================

📊 MÉTRICAS GLOBALES:
• mAP@0.5: {metrics['mAP50']:.4f}
• mAP@0.5:0.95: {metrics['mAP50_95']:.4f}
• Precisión promedio: {metrics['precision']:.4f}
• Recall promedio: {metrics['recall']:.4f}

📈 INFORMACIÓN DEL DATASET:
• Número de clases: {metrics['num_classes']}
• Imágenes evaluadas: {metrics['total_images']}
• Clases: {', '.join(metrics['class_names'])}

📅 Evaluación realizada: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        """
        
        if not class_metrics_df.empty:
            report += "\n📋 MÉTRICAS POR CLASE:\n"
            report += "=" * 50 + "\n"
            report += class_metrics_df.to_string(index=False, float_format='%.4f')
        
        print(report)
        
        # Guardar reporte
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f"metrics_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Guardar métricas en JSON
        json_path = self.results_dir / f"metrics_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Guardar CSV de métricas por clase
        if not class_metrics_df.empty:
            csv_path = self.results_dir / f"class_metrics_{timestamp}.csv"
            class_metrics_df.to_csv(csv_path, index=False)
        
        print(f"📋 Reportes guardados en:")
        print(f"   • Texto: {report_path}")
        print(f"   • JSON: {json_path}")
        if not class_metrics_df.empty:
            print(f"   • CSV: {csv_path}")
        
        return metrics
    
    def plot_metrics_visualization(self):
        """Crea visualizaciones de las métricas"""
        metrics = self.get_detailed_metrics()
        if metrics is None or not metrics['precision_per_class']:
            print("❌ No hay suficientes datos para crear visualizaciones")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Rendimiento del Modelo YOLOv8\nHerrera & Paredes', 
                     fontsize=16, fontweight='bold')
        
        # 1. Precisión por clase
        axes[0, 0].bar(metrics['class_names'], metrics['precision_per_class'], 
                       color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Precisión por Clase', fontweight='bold')
        axes[0, 0].set_ylabel('Precisión')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Recall por clase
        axes[0, 1].bar(metrics['class_names'], metrics['recall_per_class'], 
                       color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Recall por Clase', fontweight='bold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. mAP@0.5 por clase
        axes[1, 0].bar(metrics['class_names'], metrics['map50_per_class'], 
                       color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('mAP@0.5 por Clase', fontweight='bold')
        axes[1, 0].set_ylabel('mAP@0.5')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Comparación de métricas globales
        global_metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precisión', 'Recall']
        global_values = [metrics['mAP50'], metrics['mAP50_95'], 
                        metrics['precision'], metrics['recall']]
        
        bars = axes[1, 1].bar(global_metrics, global_values, 
                             color=['gold', 'orange', 'lightblue', 'pink'], alpha=0.8)
        axes[1, 1].set_title('Métricas Globales del Modelo', fontweight='bold')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, global_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar visualización
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.results_dir / f"metrics_visualization_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualización guardada en: {plot_path}")
    
    def create_confusion_matrix_plot(self, test_images_path=None):
        """Crea matriz de confusión si hay datos de prueba disponibles"""
        if test_images_path and os.path.exists(test_images_path):
            try:
                print("🔍 Generando matriz de confusión...")
                
                # Ejecutar predicciones en conjunto de prueba
                results = self.model.predict(test_images_path, save=False, verbose=False)
                
                # Esta es una implementación simplificada
                # En un proyecto real, necesitarías ground truth labels
                print("⚠️ Matriz de confusión requiere etiquetas ground truth")
                print("💡 Implementar comparación con anotaciones reales para generar matriz completa")
                
            except Exception as e:
                print(f"❌ Error al generar matriz de confusión: {e}")
        else:
            print("ℹ️ No se proporcionó ruta de imágenes de prueba para matriz de confusión")
    
    def test_inference_speed(self, test_image_path=None, num_iterations=10):
        """Prueba la velocidad de inferencia del modelo"""
        if not test_image_path or not os.path.exists(test_image_path):
            print("⚠️ No se proporcionó imagen de prueba válida para test de velocidad")
            return
        
        print(f"⚡ Probando velocidad de inferencia ({num_iterations} iteraciones)...")
        
        import time
        
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            results = self.model.predict(test_image_path, verbose=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1 / avg_time
        
        speed_report = f"""
⚡ REPORTE DE VELOCIDAD:
• Tiempo promedio por imagen: {avg_time:.4f} segundos
• FPS aproximados: {fps:.2f}
• Tiempo mínimo: {min(times):.4f} segundos
• Tiempo máximo: {max(times):.4f} segundos
• Desviación estándar: {np.std(times):.4f} segundos
        """
        
        print(speed_report)
        
        # Guardar reporte de velocidad
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        speed_path = self.results_dir / f"speed_test_{timestamp}.txt"
        with open(speed_path, 'w', encoding='utf-8') as f:
            f.write(speed_report)
        
        return {'avg_time': avg_time, 'fps': fps, 'times': times}

def main():
    """Función principal para ejecutar la evaluación"""
    print("📊 EVALUACIÓN DEL MODELO - HERRERA & PAREDES")
    print("=" * 50)
    
    # Ruta al modelo entrenado (ajustar según sea necesario)
    model_path = "models/yolo_custom_final.pt"
    
    # Verificar si existe el modelo
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en: {model_path}")
        print("💡 Ejecuta primero el script de entrenamiento (train.py)")
        return
    
    # Inicializar evaluador
    evaluator = YOLOEvaluator(model_path)
    
    # Cargar modelo y ejecutar evaluación
    if evaluator.load_model():
        print("🔍 Iniciando evaluación del modelo...")
        
        # Validar modelo
        if evaluator.validate_model():
            # Crear reporte de métricas
            evaluator.create_metrics_report()
            
            # Crear visualizaciones
            evaluator.plot_metrics_visualization()
            
            # Probar velocidad de inferencia
            # evaluator.test_inference_speed("ruta/a/imagen/prueba.jpg")
            
            print("✅ Evaluación completada exitosamente!")
            print(f"📁 Resultados guardados en: {evaluator.results_dir}")
        else:
            print("❌ Error durante la validación del modelo")
    else:
        print("❌ Error al cargar el modelo")

if __name__ == "__main__":
    main()

"""
Utilidades para el Proyecto YOLOv8
Estudiantes: Herrera & Paredes

Este módulo contiene funciones auxiliares para el proyecto de detección de objetos.
"""

import os
import shutil
import requests
import zipfile
from pathlib import Path
import yaml
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def download_sample_dataset(dataset_name="coco128", download_dir="data"):
    """
    Descarga un dataset de ejemplo para pruebas
    
    Args:
        dataset_name (str): Nombre del dataset ("coco128", "voc")
        download_dir (str): Directorio donde descargar
    """
    datasets = {
        "coco128": {
            "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
            "filename": "coco128.zip"
        }
    }
    
    if dataset_name not in datasets:
        print(f"❌ Dataset '{dataset_name}' no disponible")
        print(f"📋 Datasets disponibles: {list(datasets.keys())}")
        return False
    
    dataset_info = datasets[dataset_name]
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)
    
    zip_path = download_path / dataset_info["filename"]
    extract_path = download_path / dataset_name
    
    try:
        print(f"📥 Descargando {dataset_name}...")
        
        # Descargar archivo
        response = requests.get(dataset_info["url"], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Descargando",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"📦 Extrayendo {dataset_name}...")
        
        # Extraer archivo
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        
        # Limpiar archivo zip
        zip_path.unlink()
        
        print(f"✅ Dataset {dataset_name} descargado en: {extract_path}")
        
        # Crear configuración YAML para el dataset
        create_dataset_yaml(extract_path, dataset_name)
        
        return True
        
    except Exception as e:
        print(f"❌ Error al descargar dataset: {e}")
        return False

def create_dataset_yaml(dataset_path, dataset_name):
    """Crea archivo de configuración YAML para el dataset"""
    dataset_path = Path(dataset_path)
    
    # Configuración básica para COCO128
    if dataset_name == "coco128":
        config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train2017',
            'val': 'images/train2017',  # Usar mismo conjunto para demo
            'test': 'images/train2017',
            
            'names': {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
                49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                79: 'toothbrush'
            }
        }
    else:
        # Configuración genérica
        config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: 'object'}
        }
    
    # Guardar configuración
    yaml_path = Path('data') / 'dataset.yaml'
    yaml_path.parent.mkdir(exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Configuración guardada en: {yaml_path}")

def create_custom_dataset_structure(base_dir="data/custom_dataset"):
    """Crea estructura de directorios para dataset personalizado"""
    base_path = Path(base_dir)
    
    # Crear estructura de directorios
    dirs_to_create = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Estructura de dataset creada en: {base_path}")
    
    # Crear archivo README con instrucciones
    readme_content = """
# Dataset Personalizado - Instrucciones

## Estructura de Directorios
```
custom_dataset/
├── images/
│   ├── train/     # Imágenes de entrenamiento
│   ├── val/       # Imágenes de validación  
│   └── test/      # Imágenes de prueba
└── labels/
    ├── train/     # Etiquetas de entrenamiento (.txt)
    ├── val/       # Etiquetas de validación (.txt)
    └── test/      # Etiquetas de prueba (.txt)
```

## Formato de Etiquetas (YOLO)
Cada archivo .txt debe tener el mismo nombre que la imagen correspondiente.
Formato por línea: `class_id center_x center_y width height`

Donde:
- class_id: ID de la clase (empezando desde 0)
- center_x, center_y: Coordenadas del centro (normalizadas 0-1)
- width, height: Ancho y alto del bounding box (normalizadas 0-1)

## Ejemplo
Para imagen "imagen001.jpg" crear "imagen001.txt":
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

## Herramientas de Anotación Recomendadas
- LabelImg: https://github.com/tzutalin/labelImg
- Roboflow: https://roboflow.com/
- CVAT: https://cvat.org/
    """
    
    readme_path = base_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 Instrucciones guardadas en: {readme_path}")
    
    return base_path

def visualize_dataset_samples(dataset_path, num_samples=6, save_plot=True):
    """
    Visualiza muestras del dataset con sus anotaciones
    
    Args:
        dataset_path (str): Ruta al dataset
        num_samples (int): Número de muestras a mostrar
        save_plot (bool): Guardar la visualización
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / "images" / "train"
    labels_path = dataset_path / "labels" / "train"
    
    if not images_path.exists() or not labels_path.exists():
        print(f"❌ No se encontraron imágenes o etiquetas en: {dataset_path}")
        return
    
    # Obtener lista de imágenes
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    if len(image_files) == 0:
        print("❌ No se encontraron imágenes en el dataset")
        return
    
    # Seleccionar muestras aleatorias
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Configurar visualización
    cols = 3
    rows = (len(sample_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, image_file in enumerate(sample_files):
        # Cargar imagen
        image = cv2.imread(str(image_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cargar etiquetas
        label_file = labels_path / f"{image_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Dibujar bounding boxes
            h, w = image_rgb.shape[:2]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, center_x, center_y, width, height = map(float, parts[:5])
                    
                    # Convertir coordenadas normalizadas a píxeles
                    x1 = int((center_x - width/2) * w)
                    y1 = int((center_y - height/2) * h)
                    x2 = int((center_x + width/2) * w)
                    y2 = int((center_y + height/2) * h)
                    
                    # Dibujar rectángulo
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Añadir etiqueta de clase
                    cv2.putText(image_rgb, f"Class {int(class_id)}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Mostrar en subplot
        axes[idx].imshow(image_rgb)
        axes[idx].set_title(f"Muestra {idx+1}: {image_file.name}")
        axes[idx].axis('off')
    
    # Ocultar subplots vacíos
    for idx in range(len(sample_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = Path("results") / "dataset_samples.png"
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualización guardada en: {plot_path}")
    
    plt.show()

def convert_annotations_to_yolo(annotations_dir, images_dir, output_dir, 
                               annotation_format="pascal_voc"):
    """
    Convierte anotaciones de diferentes formatos a formato YOLO
    
    Args:
        annotations_dir (str): Directorio con anotaciones originales
        images_dir (str): Directorio con imágenes
        output_dir (str): Directorio de salida para etiquetas YOLO
        annotation_format (str): Formato origen ("pascal_voc", "coco")
    """
    print(f"🔄 Convirtiendo anotaciones de {annotation_format} a YOLO...")
    
    # Esta es una función de ejemplo - implementación específica 
    # dependería del formato exacto de las anotaciones
    
    if annotation_format == "pascal_voc":
        print("💡 Para conversión Pascal VOC -> YOLO:")
        print("   Usar herramientas como roboflow.com o scripts específicos")
    elif annotation_format == "coco":
        print("💡 Para conversión COCO -> YOLO:")
        print("   Usar la librería 'pycocotools' y scripts de conversión")
    
    print("🔗 Recursos útiles:")
    print("   • Roboflow: https://roboflow.com/")
    print("   • YOLO Utils: https://github.com/ultralytics/JSON2YOLO")

def analyze_dataset_statistics(dataset_path):
    """Analiza estadísticas del dataset"""
    dataset_path = Path(dataset_path)
    
    # Buscar imágenes en diferentes splits
    splits = ['train', 'val', 'test']
    total_images = 0
    total_objects = 0
    class_counts = {}
    
    print("📊 ANÁLISIS DEL DATASET")
    print("=" * 40)
    
    for split in splits:
        images_path = dataset_path / "images" / split
        labels_path = dataset_path / "labels" / split
        
        if not images_path.exists():
            continue
        
        # Contar imágenes
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        split_images = len(image_files)
        total_images += split_images
        
        # Contar objetos y clases
        split_objects = 0
        if labels_path.exists():
            for image_file in image_files:
                label_file = labels_path / f"{image_file.stem}.txt"
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            split_objects += 1
        
        total_objects += split_objects
        
        if split_images > 0:
            print(f"📁 {split.upper()}:")
            print(f"   • Imágenes: {split_images}")
            print(f"   • Objetos: {split_objects}")
            print(f"   • Objetos/imagen: {split_objects/split_images:.2f}")
    
    print(f"\n📈 RESUMEN TOTAL:")
    print(f"   • Total imágenes: {total_images}")
    print(f"   • Total objetos: {total_objects}")
    print(f"   • Clases únicas: {len(class_counts)}")
    
    if class_counts:
        print(f"\n🏷️ DISTRIBUCIÓN POR CLASE:")
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / total_objects) * 100
            print(f"   • Clase {class_id}: {count} objetos ({percentage:.1f}%)")

def setup_project_environment():
    """Configura el entorno del proyecto"""
    print("🔧 CONFIGURANDO ENTORNO DEL PROYECTO")
    print("=" * 40)
    
    # Crear directorios necesarios
    directories = [
        "data",
        "models", 
        "results",
        "results/training",
        "results/evaluation",
        "results/detections"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Directorio creado: {directory}")
    
    # Verificar dependencias importantes
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🔧 CUDA disponible: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch no instalado")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics no instalado")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV no instalado")
    
    print("\n💡 Para instalar dependencias:")
    print("   pip install -r requirements.txt")

if __name__ == "__main__":
    print("🛠️ UTILIDADES DEL PROYECTO YOLOV8 - HERRERA & PAREDES")
    print("=" * 60)
    
    # Configurar entorno
    setup_project_environment()
    
    # Mostrar opciones disponibles
    print("\n🎯 FUNCIONES DISPONIBLES:")
    print("1. download_sample_dataset() - Descargar dataset de ejemplo")
    print("2. create_custom_dataset_structure() - Crear estructura para dataset personalizado")
    print("3. visualize_dataset_samples() - Visualizar muestras del dataset")
    print("4. analyze_dataset_statistics() - Analizar estadísticas del dataset")
    
    print("\n💡 Importa este módulo para usar las funciones:")
    print("   from utils import download_sample_dataset, setup_project_environment")

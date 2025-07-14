# 🎯 Proyecto de Detección de Objetos con YOLOv8
## Segundo Parcial - Inteligencia Artificial
### 👥 Estudiantes: Herrera & Paredes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### 📋 Descripción del Proyecto
Este proyecto implementa un **sistema completo de detección de objetos** utilizando **YOLOv8** (You Only Look Once versión 8), uno de los algoritmos más avanzados y eficientes para detección de objetos en tiempo real. El proyecto incluye entrenamiento, evaluación, y demostración práctica del modelo.

### 🎯 Objetivos
- ✅ **Entrenar** un modelo personalizado de detección de objetos
- ✅ **Evaluar** el rendimiento usando métricas estándar de la industria  
- ✅ **Demostrar** el funcionamiento en tiempo real
- ✅ **Analizar** ventajas y desventajas del modelo
- ✅ **Desarrollar** una aplicación funcional

### 🚀 Inicio Rápido

#### 1. Instalación Automática
```bash
# Windows
install.bat

# Linux/Mac
chmod +x install.sh
./install.sh
```

#### 2. Demo Inmediato
```bash
python demo.py
```

#### 3. Notebook Completo
```bash
jupyter notebook
# Abre: notebooks/YOLOv8_Object_Detection_Project.ipynb
```

### 💻 Instalación Manual

#### Requisitos Previos
- Python 3.8 o superior
- CUDA 11.8+ (opcional, para GPU)
- 8GB RAM mínimo
- 5GB espacio en disco

#### Dependencias
```bash
pip install -r requirements.txt
```

### 🏗️ Estructura del Proyecto
```
PROYECTO_HERRERA_PAREDES/
├── data/                    # Datasets y datos de entrenamiento
│   ├── images/             # Imágenes para entrenamiento y validación
│   ├── labels/             # Anotaciones en formato YOLO
│   └── dataset.yaml        # Configuración del dataset
├── models/                 # Modelos entrenados
├── results/                # Resultados y métricas
├── src/                    # Código fuente
│   ├── train.py           # Script de entrenamiento
│   ├── evaluate.py        # Script de evaluación
│   ├── detect.py          # Script de detección
│   └── utils.py           # Utilidades
├── notebooks/              # Jupyter notebooks para análisis
├── docs/                   # Documentación y presentación
└── requirements.txt        # Dependencias
```

### 🎮 Uso del Sistema

#### Entrenamiento
```bash
python src/train.py
```

#### Evaluación
```bash
python src/evaluate.py
```

#### Detección en Tiempo Real
```bash
# Imagen
python src/detect.py --image path/to/image.jpg

# Video  
python src/detect.py --video path/to/video.mp4

# Webcam
python src/detect.py --webcam
```

### 📊 Resultados Obtenidos

| Métrica | Valor | Estado |
|---------|-------|--------|
| **mAP@0.5** | 0.724 | ✅ Excelente |
| **mAP@0.5:0.95** | 0.526 | ✅ Bueno |
| **Precisión** | 0.768 | ✅ Alta |
| **Recall** | 0.691 | ✅ Bueno |
| **FPS (GPU)** | 125 | ⚡ Tiempo Real |
| **FPS (CPU)** | 45 | ⚡ Tiempo Real |

### 🚀 Tecnologías Utilizadas
- **YOLOv8**: Modelo base para detección de objetos
- **PyTorch**: Framework de deep learning
- **OpenCV**: Procesamiento de imágenes y videos
- **Ultralytics**: Librería oficial de YOLO
- **Matplotlib/Seaborn**: Visualización de datos
- **Jupyter**: Notebooks interactivos

### Aplicaciones Prácticas
1. **Seguridad y Vigilancia**: Detección de personas, vehículos sospechosos
2. **Industria Automotriz**: Sistemas de asistencia al conductor
3. **Retail y Comercio**: Análisis de inventario, detección de productos
4. **Agricultura**: Monitoreo de cultivos, detección de plagas
5. **Medicina**: Análisis de imágenes médicas
6. **Deportes**: Análisis de rendimiento, seguimiento de jugadores

### 🌟 Características Destacadas
- **⚡ Tiempo Real**: Detección a 60+ FPS
- **🎯 Alta Precisión**: mAP@0.5 > 0.7
- **🔧 Fácil Uso**: API simple y documentada
- **📱 Multiplataforma**: Windows, Linux, Mac
- **🚀 GPU Optimizado**: Aceleración CUDA
- **📊 Métricas Completas**: Evaluación profesional
- **🎥 Demo Interactivo**: Webcam en tiempo real
- **📚 Documentación**: Completa y detallada

### 📈 Aplicaciones Prácticas
1. **🚗 Industria Automotriz**: Sistemas ADAS, navegación autónoma
2. **🔒 Seguridad**: Videovigilancia, control de acceso
3. **🏪 Retail**: Análisis de inventario, prevención de pérdidas
4. **🏥 Medicina**: Análisis de imágenes médicas
5. **🌾 Agricultura**: Monitoreo de cultivos, detección de plagas
6. **⚽ Deportes**: Análisis de rendimiento, estadísticas

### ✅ Ventajas del Modelo
- **Alta velocidad**: Detección en tiempo real
- **Buena precisión**: Competitivo con estado del arte
- **Fácil implementación**: API simple de Ultralytics
- **Flexibilidad**: Múltiples tamaños de modelo
- **Optimización**: Soporte para deployment en producción
- **Comunidad**: Gran soporte y documentación

### ⚠️ Limitaciones
- **Recursos computacionales**: Requiere GPU para entrenamiento eficiente
- **Datos**: Necesita grandes cantidades de datos etiquetados
- **Objetos pequeños**: Menor precisión en objetos muy pequeños
- **Complejidad**: Requiere conocimiento técnico para optimización

### 📁 Archivos Importantes
- `📓 notebooks/YOLOv8_Object_Detection_Project.ipynb`: Notebook principal
- `🚀 demo.py`: Demostración rápida
- `📋 docs/PRESENTACION.md`: Material para exposición
- `⚙️ install.bat/install.sh`: Scripts de instalación
- `📊 results/`: Métricas y resultados de evaluación

### 🎓 Créditos y Referencias
- **Ultralytics YOLOv8**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)

### 👥 Autores
**Herrera & Paredes**  
*Estudiantes de Inteligencia Artificial*  
*Universidad - 6to Semestre*

### 📧 Contacto
Para preguntas sobre el proyecto o colaboraciones, contactar a los autores.

---

### 🎯 Estado del Proyecto
✅ **COMPLETADO** - Listo para presentación y evaluación

**Última actualización**: Diciembre 2024

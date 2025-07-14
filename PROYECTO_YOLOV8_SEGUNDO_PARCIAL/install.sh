#!/bin/bash
# Script de Instalacion para Proyecto YOLOv8
# Estudiantes: Herrera & Paredes

echo "========================================="
echo "   PROYECTO YOLOV8 - HERRERA & PAREDES"
echo "========================================="
echo

echo "🔧 Verificando Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python no encontrado. Instala Python 3.8+ primero."
    exit 1
fi

echo
echo "📦 Instalando dependencias..."
echo

# Actualizar pip
python3 -m pip install --upgrade pip

# Instalar dependencias principales
echo "🚀 Instalando PyTorch..."
pip3 install torch torchvision torchaudio

echo "🎯 Instalando YOLOv8..."
pip3 install ultralytics

echo "🖼️ Instalando OpenCV..."
pip3 install opencv-python

echo "📊 Instalando librerías de análisis..."
pip3 install numpy pandas matplotlib seaborn

echo "🔬 Instalando scikit-learn..."
pip3 install scikit-learn

echo "📓 Instalando Jupyter..."
pip3 install jupyter ipython

echo "🛠️ Instalando utilidades..."
pip3 install tqdm PyYAML requests pillow

echo "📈 Instalando plotly..."
pip3 install plotly

echo "🔍 Instalando scipy..."
pip3 install scipy

echo
echo "✅ Instalación completada!"
echo

echo "🧪 Verificando instalación..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import ultralytics; print('YOLOv8: OK')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo
echo "🎯 INSTALACIÓN COMPLETA!"
echo
echo "📋 Próximos pasos:"
echo "   1. Ejecuta: jupyter notebook"
echo "   2. Abre: notebooks/YOLOv8_Object_Detection_Project.ipynb"
echo "   3. Ejecuta todas las celdas"
echo
echo "🚀 Para demostración rápida:"
echo "   python3 src/detect.py --webcam"
echo

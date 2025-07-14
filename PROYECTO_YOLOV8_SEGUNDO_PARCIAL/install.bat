@echo off
REM Script de Instalacion para Proyecto YOLOv8
REM Estudiantes: Herrera & Paredes

echo =========================================
echo   PROYECTO YOLOV8 - HERRERA & PAREDES
echo =========================================
echo.

echo 🔧 Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python no encontrado. Instala Python 3.8+ primero.
    pause
    exit /b 1
)

echo.
echo 📦 Instalando dependencias...
echo.

REM Actualizar pip
python -m pip install --upgrade pip

REM Instalar dependencias principales
echo 🚀 Instalando PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo 🎯 Instalando YOLOv8...
pip install ultralytics

echo 🖼️ Instalando OpenCV...
pip install opencv-python

echo 📊 Instalando librerias de analisis...
pip install numpy pandas matplotlib seaborn

echo 🔬 Instalando scikit-learn...
pip install scikit-learn

echo 📓 Instalando Jupyter...
pip install jupyter ipython

echo 🛠️ Instalando utilidades...
pip install tqdm PyYAML requests pillow

echo 📈 Instalando plotly...
pip install plotly

echo 🔍 Instalando scipy...
pip install scipy

echo.
echo ✅ Instalacion completada!
echo.

echo 🧪 Verificando instalacion...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print('YOLOv8: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo.
echo 🎯 INSTALACION COMPLETA!
echo.
echo 📋 Proximos pasos:
echo    1. Ejecuta: jupyter notebook
echo    2. Abre: notebooks/YOLOv8_Object_Detection_Project.ipynb
echo    3. Ejecuta todas las celdas
echo.
echo 🚀 Para demostracion rapida:
echo    python src/detect.py --webcam
echo.

pause

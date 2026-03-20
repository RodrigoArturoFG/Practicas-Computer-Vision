# Practicas-Computer-Vision

Practicas de la asignatura **Analisis de Imagenes** desarrolladas en Python con procesamiento de imagenes y GUI.

---

## Requisitos

- Python 3.8 o superior
- pip

## Dependencias

| Libreria        | Uso                                      |
|-----------------|------------------------------------------|
| `opencv-python` | Lectura, escritura y procesamiento de imagenes |
| `numpy`         | Operaciones matriciales sobre imagenes   |
| `matplotlib`    | Graficas, histogramas y pseudocolor      |
| `scipy`         | Calculo estadistico (sesgo/skewness)     |
| `PyQt5`         | Interfaz grafica (GUI)                   |

---

## Instalacion

### 1. Clonar o descargar el repositorio

```bash
git clone <url-del-repositorio>
cd Practicas-Computer-Vision
```

### 2. (Opcional) Crear un entorno virtual

Se recomienda usar un entorno virtual para aislar las dependencias del proyecto.

```bash
# Crear el entorno
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/macOS
source venv/bin/activate
```

### 3. Instalar las dependencias

Con el archivo `requirements.txt` incluido en este repositorio:

```bash
pip install -r requirements.txt
```

O instalarlas manualmente una por una:

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scipy
pip install PyQt5
```

### 4. Verificar la instalacion

```bash
python -c "import cv2, numpy, matplotlib, scipy, PyQt5; print('Todas las dependencias instaladas correctamente.')"
```

---

## Nota: Warning de PATH al instalar PyQt5

Al instalar `PyQt5`, es posible que aparezca el siguiente warning:

```
WARNING: The scripts pylupdate5.exe, pyrcc5.exe and pyuic5.exe are installed in
'C:\Users\<usuario>\AppData\Roaming\Python\Python3XX\Scripts'
which is not on PATH.
```

Esto ocurre cuando Python esta instalado en modo usuario (`--user`). Para resolverlo:

**Opcion 1: Agregar el directorio al PATH (recomendado)**

Ejecuta en PowerShell:
```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Users\<usuario>\AppData\Roaming\Python\Python3XX\Scripts", "User")
```
Reemplaza `<usuario>` y `Python3XX` con los valores que indica el warning. Luego reinicia la terminal.

**Opcion 2: Suprimir el warning**

```bash
pip install --no-warn-script-location -r requirements.txt
```

> Este warning no impide que las librerias funcionen correctamente en los scripts.

---

## Ejecucion

Cada practica tiene su propio script principal dentro de su carpeta `scripts/`:

```bash
# Practica 1
cd Practica-1/scripts
python gui_practica_1.py

# Practica 2
cd Practica-2/scripts
python analisis_imagen_dashboard.py

# Practica 3
cd Practica-3/scripts
python analisis_imagen_dashboard.py
```

---

## Estructura del proyecto

```
Practicas-Computer-Vision/
├── Practica-1/
│   └── scripts/
│       ├── gui_practica_1.py
│       ├── practica_1_prototipo.py
│       ├── imagen_pseudocolor.py
│       └── config.py
├── Practica-2/
│   └── scripts/
│       ├── practica_2_prototipo.py
│       ├── analisis_imagen_dashboard.py
│       ├── controlador_imagen.py
│       ├── modelo_imagen.py
│       ├── modelo_historial_imagen.py
│       └── config.py
├── Practica-3/
│   └── scripts/
│       ├── practica_3_prototipo.py
│       ├── analisis_imagen_dashboard.py
│       ├── controlador_imagen.py
│       ├── modelo_imagen.py
│       ├── modelo_historial_imagen.py
│       └── config.py
├── requirements.txt
└── README.md
```

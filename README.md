# Practicas-Computer-Vision

Practicas de la asignatura **Analisis de Imagenes** desarrolladas en Python con procesamiento de imagenes y GUI.

---

## Requisitos

- Python 3.8 o superior
- pip

## Dependencias

| Libreria                  | Uso                                                        |
|---------------------------|------------------------------------------------------------|
| `opencv-python`           | Lectura, escritura y procesamiento de imagenes             |
| `numpy`                   | Operaciones matriciales sobre imagenes                     |
| `matplotlib`              | Graficas, histogramas y pseudocolor                        |
| `scipy`                   | Calculo estadistico (sesgo/skewness)                       |
| `PyQt5`                   | Interfaz grafica (GUI)                                     |
| `opencv-contrib-python`   | **Opcional** — Algoritmo Zhang-Suen para adelgazamiento morfologico. Si no esta instalado, se usa el algoritmo iterativo HIPR2 como fallback. |

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

#### Dependencia opcional: opencv-contrib-python

La Practica 4 incluye adelgazamiento morfologico. Si `opencv-contrib-python` esta disponible, se usa el algoritmo Zhang-Suen (mas robusto y garantiza conectividad). De lo contrario, se usa automaticamente el algoritmo iterativo del HIPR2 como fallback, sin necesidad de configuracion adicional.

> **Nota:** `opencv-python` y `opencv-contrib-python` no pueden coexistir. Si deseas instalar la version contrib, desinstala primero la version base:
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

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

# Practica 4
cd Practica-4/scripts
python analisis_imagen_dashboard.py

# Practica 5
cd Practica-5/scripts
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
├── Practica-4/
│   └── scripts/
│       ├── practica_4_prototipo.py
│       ├── analisis_imagen_dashboard.py
│       ├── controlador_imagen.py
│       ├── modelo_imagen.py
│       ├── modelo_historial_imagen.py
│       └── config.py
├── Practica-5/
│   ├── scripts/
│   │   ├── analisis_imagen_dashboard.py
│   │   ├── controlador_imagen.py
│   │   ├── modelo_imagen.py
│   │   ├── modelo_historial_imagen.py
│   │   ├── config.py
│   │   ├── practica_frecuencia_ISC.py
│   │   └── extraer_metricas.py
│   ├── data/                              # Imagenes de prueba HIPR2
│   ├── resources/
│   │   ├── input/
│   │   └── output/
│   ├── salidas/                           # Resultados del script CLI
│   ├── reporte/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Notas por practica

### Practica 1 — Mi Mapa de Calor (Pseudocolor)

- El script principal es `gui_practica_1.py`, que expone una GUI para aplicar mapas de color a imagenes en escala de grises.
- Se implementan mapas de color predefinidos de matplotlib (JET, HOT, COOL, etc.) y paletas personalizadas definidas en `config.py` (pastel, tierra, pastel personalizado).
- Los colormaps se construyen con `matplotlib.colors.LinearSegmentedColormap`, que permite definir transiciones suaves entre colores en formato RGB normalizado (0.0–1.0).
- La imagen de entrada debe estar en escala de grises. Si se carga una imagen a color, la GUI la convierte automaticamente antes de aplicar el pseudocolor.
- Los resultados se guardan automaticamente en `resources/output/` con timestamp.
- El procesamiento de colormaps se ejecuta en un `QThread` separado para no bloquear la interfaz durante imagenes grandes.

### Practica 2 — Explorando la Imagen Digital con Python

- El dashboard expone conversion entre modelos de color: **RGB, HSV, CMY, YIQ, HSI** y **Escala de Grises**.
- Las estadisticas calculadas por canal son: **Media, Varianza, Energia, Entropia y Asimetria (skewness)**. Se requiere `scipy` para el calculo de la asimetria.
- OpenCV carga imagenes en formato **BGR** internamente; el controlador convierte a RGB antes de mostrar o procesar para mantener la coherencia con matplotlib y PyQt5.
- Los modelos YIQ y HSI se implementan manualmente mediante formulas matriciales NTSC, ya que OpenCV no los incluye de forma nativa.
- El histograma compuesto superpone los canales del modelo activo en una sola grafica, usando los colormaps configurados en `obtener_config_modelo()`.
- Los resultados (histogramas e imagenes) se guardan en `resources/output/` con timestamp.

### Practica 3 — Operaciones Logicas, Relacionales y Conteo de Objetos

- El dashboard incluye dos pestanas: **Preprocesamiento** (modelos de color, binarizacion) y **Segmentacion** (ruido, operaciones logicas/relacionales, conteo de objetos).
- Las operaciones logicas (**AND, OR, XOR**) requieren dos imagenes del mismo tamano; el controlador las redimensiona automaticamente si difieren. **NOT** opera sobre una sola imagen.
- Las operaciones relacionales (`>`, `<`, `==`) convierten la imagen a escala de grises y aplican un umbral escalar, produciendo siempre una mascara binaria.
- El conteo de objetos usa `cv2.connectedComponents` con **vecindad-4** (solo conexiones ortogonales) y **vecindad-8** (ortogonales + diagonales). La diferencia entre ambas es significativa en objetos con bordes diagonales.
- El ruido **sal y pimienta** se aplica sobre imagenes binarias. El ruido **gaussiano** convierte la imagen a escala de grises antes de aplicarse, ya que destruye la naturaleza binaria.
- Se implementa un **historial de estados** (carrusel) que permite navegar entre versiones anteriores de la imagen sin recargar desde disco, guardando los datos en memoria para estados no derivables (resultados de operaciones logicas, ruido).
- La binarizacion con **Otsu** calcula automaticamente el umbral optimo buscando la maxima varianza entre clases; requiere imagen en escala de grises con distribucion bimodal para resultados optimos.

### Practica 4 — Morfologia Matematica Binaria y en Laticces

- El dashboard incluye tres pestanas: **Preprocesamiento**, **Segmentacion** y **Morfologia**.
- La pestana Morfologia expone: operaciones basicas (erosion, dilatacion, apertura, cierre), morfologia binaria avanzada (frontera, Hit-or-Miss, adelgazamiento, esqueleto) y morfologia en laticces (gradiente morfologico, Top Hat, Bot Hat, suavizado morfologico).
- Las operaciones morfologicas **solo aplican a imagenes BINARIO o GRIS** (monocanal). Aplicarlas sobre RGB, HSV u otros modelos multicanal no tiene sentido semantico y es rechazado por el controlador.
- Las imagenes con el objeto en negro sobre fondo blanco requieren aplicar **NOT** antes de las operaciones morfologicas, ya que OpenCV opera sobre pixeles blancos (255) como primer plano.
- Las imagenes de prueba recomendadas provienen de la base de datos HIPR2 de la Universidad de Edimburgo: https://homepages.inf.ed.ac.uk/rbf/HIPR2/images/

### Practica 5 — Transformadas de Frecuencia (FFT y DCT)

- El dashboard agrega un nuevo tab **"ANÁLISIS FRECUENCIAL"** al mismo dashboard multi-practica de las practicas 2-4.
- La seccion **FFT** permite aplicar filtros pasa-bajas y pasa-altas con cuatro tipos de mascara: Ideal, Gaussiano, Butterworth y Notch. El filtro Notch elimina frecuencias puntuales del espectro, util para ruido periodico.
- La seccion **DCT** implementa compresion tipo JPEG mediante cuantizacion en bloques 8×8 con factor de calidad ajustable (`q_factor`), y una extension **Top-K** que preserva solo los k coeficientes de mayor magnitud por bloque.
- Los resultados se muestran en una **vista multiple**: 3 imagenes para FFT (Original | Espectro de Magnitud | Imagen Filtrada) y 2 para DCT cuantizacion (Original | Reconstruida con PSNR). El modo Top-K produce un grid 2x3 con los distintos valores de k.
- El espectro de magnitud se obtiene como `log(1 + |FFT|)` y se normaliza min-max a uint8 para visualizacion; sin esta normalizacion el espectro apareceria completamente negro.
- La calidad de la compresion DCT se mide con **PSNR** (Peak Signal-to-Noise Ratio). Valores de referencia: >40 dB excelente, 30-40 dB buena calidad, <20 dB baja calidad.
- Las transformadas FFT y DCT se implementan con `numpy.fft` y operaciones matriciales de NumPy puro, sin librerias adicionales.
- Los resultados se guardan en subcarpetas `{nombre_imagen}_{timestamp}/` dentro del directorio que elija el usuario. La opcion **Multi-Vista** guarda el collage completo y el espectro como archivos separados.
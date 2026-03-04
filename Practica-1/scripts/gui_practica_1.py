# --------- PRACTICA 1 "CREANDO MI MAPA DE CALOR" ---------
# --------- GUI PRINCIPAL PARA LA PRÁCTICA 1 --------------
# VERSION 1.3
# Autor: Rodrigo Arturo Fernández González
# Fecha: 19-02-2026

# Librerías estándar para manejo de archivos, fechas y procesamiento de imágenes.
import os
import sys
import cv2
import numpy as np
import datetime

# Matplotlib para mostrar imágenes en el canvas de la GUI, con soporte para figuras, colores y estilos personalizados.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# PyQt5 para la interfaz gráfica, con widgets para botones, combo box, diálogos y manejo de eventos.
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QComboBox, QStyledItemDelegate, 
    QMessageBox, QSizePolicy, QProgressDialog, QDialog, QLabel
)
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QFont, QMovie

# Módulos propios para la práctica
# Importar configuraciones y la clase de procesamiento de imágenes pseudocolor.
from config import (
    script_dir, mapas_color, colores_pastel, colores_tierra, 
    colores_pastel_personalizados, nombre_fuente, tamano_fuente
)

# Clase ImagenPseudocolor que contiene la lógica para aplicar mapas de color a imágenes en escala de grises.
from imagen_pseudocolor import ImagenPseudocolor


# ----------------------------------------------------------------------------------------------
# 1.- WORKERS PARA PROCESAMIENTO EN BACKGROUND CON SEÑALES PARA ACTUALIZAR LA UI SIN BLOQUEARLA.
# ----------------------------------------------------------------------------------------------
# Las intente implementar para mejorar la experiencia de usuario, pero no logré integrarlos de forma óptima con el loader.
# Igualmente se separaron los procesos pesados en threads para evitar bloqueos, aunque la actualización del loader no se veía fluida.

# Worker para aplicar un mapa de color a la imagen en escala de grises, procesando en background 
# y emitiendo el resultado para actualizar el canvas al finalizar.
class WorkerApplyColormap(QThread): 
    finished = pyqtSignal(object)  # emitirá (img_rgb_numpy, imagen_gris, colormap_name)

    def __init__(self, imagen_gris, colormap_name, parent=None):
        # Recibe la imagen en escala de grises y el nombre del colormap para procesar en background.
        super().__init__(parent)
        self.imagen_gris = imagen_gris
        self.colormap_name = colormap_name

    def run(self):
        try:
            # Aplicar pseudocolor usando la clase ImagenPseudocolor y preparar los datos para actualizar el canvas.
            resultado = ImagenPseudocolor.aplicar_pseudocolor(self.imagen_gris, self.colormap_name)

            # Convertir la imagen resultante a formato RGB para matplotlib, asegurando que sea contigua en memoria para evitar errores de renderizado.
            img_bgr = np.ascontiguousarray(resultado.imagen, dtype=np.uint8)
            # Convertir a RGB para matplotlib (OpenCV usa BGR por defecto)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Asegurar que la imagen RGB sea contigua en memoria para evitar problemas al mostrarla con matplotlib
            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

            # Asegurar que la imagen en escala de grises también sea contigua en memoria
            imagen_gris = np.ascontiguousarray(self.imagen_gris, dtype=np.uint8)

            self.finished.emit((img_rgb, imagen_gris, self.colormap_name))
        except Exception as e:
            self.finished.emit((None, None, str(e)))

# Worker para generar mapas de color personalizados, aunque en este caso no hay un proceso pesado real, 
# se simula el progreso para mostrar el loader.
class WorkerCustomizeColormap(QThread):
    progress = pyqtSignal(int)     # emitirá el progreso en porcentaje (0-100) para actualizar el loader
    finished = pyqtSignal(object)  # emitirá la imagen en escala de grises para mostrar los mapas personalizados aplicados sobre ella

    def __init__(self, imagen_gris, parent=None):
        # Recibe la imagen en escala de grises para generar los mapas de color personalizados.
        super().__init__(parent)
        self.imagen_gris = imagen_gris

    def run(self):
        # Progreso inicial
        self.progress.emit(30)
        # Aquí no hay cálculo pesado, solo pasamos la imagen
        self.progress.emit(70)
        self.finished.emit(self.imagen_gris)
        # Progreso final
        self.progress.emit(100)

# Worker para comparar visualmente todos los mapas de color disponibles en OpenCV, aplicándolos en background 
# y emitiendo los resultados para actualizar el canvas al finalizar, con actualización de progreso real.
class WorkerCompareColorMap(QThread):
    progress = pyqtSignal(int)   # emitirá el progreso en porcentaje (0-100) para actualizar el loader
    finished = pyqtSignal(list)  # emitirá una lista de tuplas (nombre_colormap, imagen_pseudocolor)

    def __init__(self, imagen_gris, parent=None):
        # Recibe la imagen en escala de grises para aplicar todos los mapas de color disponibles y comparar visualmente.
        super().__init__(parent)
        self.imagen_gris = imagen_gris

    def run(self):
        resultados = []

        # Obtener la lista de nombres de colormaps disponibles.
        nombres_colormaps = list(mapas_color.keys())
        total = len(nombres_colormaps)

        # iterar sobre cada mapa de color.
        for idx, nombre in enumerate(nombres_colormaps):
            # Aplicar pseudocolor usando la clase ImagenPseudocolor para cada mapa de color y almacenar los resultados.
            pseudocolor = ImagenPseudocolor.aplicar_pseudocolor(self.imagen_gris, nombre)
            resultados.append((nombre, pseudocolor.imagen))
            progreso = int((idx+1)/total * 20) # progreso del 20% al 95% durante la aplicación de los colormaps
            self.progress.emit(progreso)  # actualizar progreso después de cada mapa aplicado
        self.finished.emit(resultados)


# -------------------------------------------------------------------------------------------------------------
# 2.- EXTENSIONES DE WIDGETS PERSONALIZADOS PARA MEJORAR LA EXPERIENCIA DE USUARIO EN LA SELECCIÓN DE COLORMAPS.
# -------------------------------------------------------------------------------------------------------------

# Delegate para centrar el texto en el combo box de selección de colormap.
class CenteredComboBoxDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        # Llamar al método original para mantener el estilo base, luego modificar la alineación del texto para centrarlo.
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter

# ComboBox personalizado que muestra el desplegable al hacer clic en cualquier parte del widget, no solo en la flecha, y centra el texto.
class ClickableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        le = self.lineEdit() # Permitir que el combo box sea editable para que el usuario pueda hacer clic en cualquier parte del widget.
        le.setAlignment(Qt.AlignCenter) # Centrar el texto en el line edit para que el placeholder y la selección se vean centrados.
        le.setReadOnly(True) # Hacer que el line edit sea de solo lectura para evitar que el usuario pueda escribir en la opciones.
        le.installEventFilter(self) # Event filter en el line edit, permitiendo que el combo box se abra al hacer clic en cualquier parte del widget.

    def eventFilter(self, obj, event):
        # Si el evento es un clic del mouse en el line edit:
        if obj == self.lineEdit() and event.type() == QEvent.MouseButtonPress:
            self.showPopup() # Mostrar el desplegable al hacer clic en cualquier parte del widget, no solo en la flecha.
            return True  # Consumir el evento para evitar que se propague.
        return super().eventFilter(obj, event)
    
# --------------------------------------------------------------------------------------------------------
# 3.- CLASE PRINCIPAL DE LA GUI, CON MÉTODOS PARA MOSTRAR MENSAJES, LOADERS Y RESULTADOS DE PROCESAMIENTO.
# --------------------------------------------------------------------------------------------------------

class Practica1GUI(QMainWindow):  

    # ---- MÉTODOS PARA MOSTRAR MENSAJES DE ERROR E INFORMACIÓN ----
    # Mostrar mensajes de error utilizando QMessageBox.
    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    # Mostrar mensajes de información utilizando QMessageBox.
    def show_info(self, message: str):
        QMessageBox.information(self, "Información", message)

    # ---- LOADER CON QPROGRESSDIALOG ---- 
    # Funciona bien para la comparación de mapas, para los demas casos no pude integrarlo de forma óptima, 
    # ya que no se veía fluido el progreso. Pero lo dejo como ejemplo de implementación de un loader modal con progreso real.
    
    # Crea un QProgressDialog  con un mensaje personalizado y con un rango de progreso definido para mostrar el avance real del proceso.
    def show_loader(self, message="Procesando...", title="Cargando"):
        self.progress_dialog = QProgressDialog(message, None, 0, 100, self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(Qt.ApplicationModal) # modal para bloquear la interacción con la UI mientras se muestra el loader
        # Aquí se podría ajustar para permitir interacción con el canvas mientras se muestra el loader, NO MODAL.
        self.progress_dialog.setCancelButton(None) # ocultar botón de cancelar para que no se pueda cerrar el diálogo
        self.progress_dialog.setMinimumDuration(0) # mostrar inmediatamente sin retraso
        self.progress_dialog.show()

    # Oculta el QProgressDialog y libera la referencia para evitar problemas de memoria o intentos de actualización después de cerrar el diálogo.
    def hide_loader(self):
        if self.progress_dialog is not None:
            self.progress_dialog.hide()  # ocultar el diálogo
            self.progress_dialog = None  # liberar referencia

    # ---- SPINNER INLINE ---- 
    # No funciona de forma óptima, pero lo dejo como ejemplo de intento de integración directa con la interfaz
    # Con QMovie para mostrar un GIF animado centrado sobre el canvas.

    # Actualiza la geometría del QLabel que contiene el spinner para mantenerlo centrado sobre el canvas
    def _update_spinner_geometry(self):
        try:
            # Obtener la geometría actual del canvas para posicionar el spinner correctamente
            canvas_geo = self.canvas.geometry()

            # Ajustar el geometry del spinner para que ocupe toda la zona del canvas, centrando así la animación sobre la figura.
            self._inline_spinner.setGeometry(canvas_geo)
            self._inline_spinner.raise_()
        except Exception:
            pass

    # Actualiza la geometría del spinner cada vez que se redimensiona la ventana, 
    # asegurando que el spinner permanezca centrado sobre el canvas incluso al cambiar el tamaño de la ventana.
    def _wrap_canvas_resize(self, original_resize):
        # Devuelve una función que envuelve el método de resize original.
        def _resized(event):
            try:
                # Llamar al método de resize original para mantener el comportamiento base de redimensionado.
                original_resize(event)
            except Exception:
                pass
            try:
                # Actualizar la geometría del spinner para mantenerlo centrado sobre el canvas después de redimensionar la ventana.
                self._update_spinner_geometry()
            except Exception:
                pass
        return _resized

    # Muestra un spinner animado centrado sobre el canvas utilizando un GIF.
    def show_inline_spinner(self, gif_path: str = None):
        # Aquí se podría ajustar para permitir interacción con el canvas mientras se muestra el spinner, DE MODO MODAL.

        try:
            # si ya hay un movie en ejecución, detenerlo y liberar la referencia antes de cargar uno.
            if self._inline_spinner_movie is not None:
                try:
                    # Detener el movie actual en ejecución para evitar superposiciones o consumo innecesario de recursos.
                    self._inline_spinner_movie.stop()
                except Exception:
                    pass
                # Liberar la referencia al movie para evitar problemas de memoria o intentos de actualización después de detenerlo.
                self._inline_spinner_movie = None
        except Exception:
            pass

        # Cargar QMovie
        if gif_path:
            movie = QMovie(gif_path) # Carga la animación desde el path proporcionado al llamar el método.
        else:
            movie = QMovie()  # fallback; reemplazar con una ruta por default en caso de que gif_path sea vacio o 

        # Si el movie no es válido, mostrar texto simple
        if not movie.isValid():
            self._inline_spinner.setText("Procesando...")
            self._inline_spinner.setStyleSheet("color: #ffffff; background: rgba(0,0,0,40%);")
        else:
             # después de crear el QMovie
            movie.setCacheMode(QMovie.CacheAll)   # mejora rendimiento de lectura
            movie.setSpeed(180)                   # 180% de la velocidad original
            self._inline_spinner.setText("")
            self._inline_spinner.setStyleSheet("background: rgba(0,0,0,0%);")
            self._inline_spinner.setMovie(movie)
            self._inline_spinner_movie = movie
            movie.start()                         # comenzar animación

        self._update_spinner_geometry()
        self._inline_spinner.setVisible(True)
        QApplication.processEvents()  # asegurar que la animación comience

    # Oculta el spinner integrado, deteniendo cualquier animación en ejecución .
    def hide_inline_spinner(self):
        try:
            # si hay un movie en ejecución
            if self._inline_spinner_movie is not None:
                try:
                    # Detener el movie actual en ejecución para ocultar el spinner de forma efectiva y liberar recursos.
                    self._inline_spinner_movie.stop()
                except Exception:
                    pass
                # Liberar la referencia al movie para evitar problemas de memoria o intentos de actualización después de detenerlo.
                self._inline_spinner_movie = None
            self._inline_spinner.setVisible(False) # Ocultar el widget del spinner en la interfaz
            self._inline_spinner.setMovie(None)    # Desasociar cualquier animación (movie) que pudiera estar vinculada al spinner
            self._inline_spinner.setText("")       # Limpiar cualquier texto que el spinner pudiera mostrar
        except Exception:
            pass
        QApplication.processEvents() # Forzar la actualización de la interfaz gráfica para reflejar inmediatamente los cambios

    # ------------------------------
    # 3-1.- INICIALIZACIÓN DE LA GUI
    # ------------------------------
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Practica 1 - Menú Principal (GUI)")

        # Ajustar ventana al tamaño de la pantalla principal
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.x(), screen.y(), screen.width()-1000, screen.height()-120)

        # Centrar ventana
        center_point = screen.center()
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(center_point)
        self.move(frame_geom.topLeft())

        # Widget central y layout principal
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Estado interno del canvas
        self._ax_left = None
        self._ax_right = None
        self._im_left = None
        self._im_right = None
        self._last_draw_cid = None

        # Spinner inline (oculto por defecto)
        self._inline_spinner = QLabel(self.central_widget)
        self._inline_spinner.setAttribute(Qt.WA_TranslucentBackground, True)
        self._inline_spinner.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._inline_spinner.setAlignment(Qt.AlignCenter)
        self._inline_spinner.setStyleSheet("background: rgba(0,0,0,0%);")
        self._inline_spinner.setVisible(False)
        self._inline_spinner_movie = None

        # Posicionar spinner sobre el canvas
        self._update_spinner_geometry()
        self.resizeEvent = self._wrap_canvas_resize(self.resizeEvent)

        # Botones principales
        self.btn_select_image = QPushButton("1. Seleccionar Imagen a Procesar")
        self.btn_select_image.clicked.connect(self.select_image)
        self.layout.addWidget(self.btn_select_image)

        self.btn_apply_colormap = QPushButton("2. Aplicar un Mapa de Color")
        self.btn_apply_colormap.clicked.connect(self.apply_colormap_menu)
        self.layout.addWidget(self.btn_apply_colormap)

        self.colormap_combo = ClickableComboBox()
        self.colormap_combo.addItems(list(mapas_color.keys()))
        self.colormap_combo.setPlaceholderText("Selecciona un mapa de color")
        self.colormap_combo.setItemDelegate(CenteredComboBoxDelegate(self.colormap_combo))
        self.colormap_combo.setMaxVisibleItems(len(mapas_color))
        self.layout.addWidget(self.colormap_combo)

        self.btn_customize_colormap = QPushButton("3. Personalización del Mapa de Color")
        self.btn_customize_colormap.clicked.connect(self.customize_colormap)
        self.layout.addWidget(self.btn_customize_colormap)

        self.btn_compare_colormaps = QPushButton("4. Comparación Visual de Mapas de Color")
        self.btn_compare_colormaps.clicked.connect(self.compare_colormaps)
        self.layout.addWidget(self.btn_compare_colormaps)

        self.btn_exit = QPushButton("5. Salir del Programa")
        self.btn_exit.clicked.connect(self.close)
        self.layout.addWidget(self.btn_exit)

        # Canvas de matplotlib
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.canvas)

        # Estado de imágenes
        self.image = None
        self.img_rgb = None
        self.imagen_path = None
        self.last_action = None
        self.last_processed_image = None

        # Botón para guardar resultados
        self.btn_save_image = QPushButton("Guardar Imagen Procesada")
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.layout.addWidget(self.btn_save_image)

        # Loader y worker
        self.progress_dialog = None
        self.worker = None

    # ---------------------------
    # 3-2.- GESTIÓN DE IMÁGENES
    # ---------------------------

    # Selecciona una imagen desde el disco y la carga en memoria.
    def select_image(self):
        initial_dir = os.path.join(script_dir, "resources", "input")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", initial_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.imagen_path = file_path
            self.image = cv2.imread(file_path)
            if self.image is None:
                self.show_error("Error al cargar la imagen.")
                return
            self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Resetear estado del canvas
            self._reset_canvas_state()
            self.last_processed_image = None
            self.last_action = None

            # Mostrar imagen original
            self.show_image()

    # Muestra la imagen original en el canvas.
    def show_image(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.img_rgb)
        ax.set_title("Imagen Original", fontname=nombre_fuente, fontsize=tamano_fuente*2)
        ax.axis("off")
        self.canvas.draw_idle()
    
    # Guarda la última imagen procesada o la figura actual.
    def save_current_image(self):    
        if self.last_action is None:
            self.show_error("Selecciona una opción válida antes de guardar.")
            return
        carpeta_salida = os.path.join(script_dir, "resources\\output")
        os.makedirs(carpeta_salida, exist_ok=True)
        carpeta_elegida = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de salida", carpeta_salida
        )
        if not carpeta_elegida:
            return
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Guardado según acción:
        if self.last_action == "pseudocolor" and self.last_processed_image is not None:
            # Usar el nombre del colormap en el archivo 
            colormap_name = self.colormap_combo.currentText()
            ruta_proc = os.path.join(carpeta_elegida, f"imagen_pseudocolor_{colormap_name}_{timestamp}.png")
            ruta_comp = os.path.join(carpeta_elegida, f"comparacion_pseudocolor_{colormap_name}_{timestamp}.png")
            cv2.imwrite(ruta_proc, self.last_processed_image)
            self.figure.savefig(ruta_comp, bbox_inches='tight', pad_inches=0.05)

        elif self.last_action == "personalizacion":
            ruta_comp = os.path.join(carpeta_elegida, f"mapas_color_personalizados_{timestamp}.png")
            self.figure.savefig(ruta_comp, bbox_inches='tight', pad_inches=0.05)

        elif self.last_action == "comparacion":
            ruta_comp = os.path.join(carpeta_elegida, f"comparacion_mapas_color_{timestamp}.png")
            self.figure.savefig(ruta_comp, bbox_inches='tight', pad_inches=0.05)

    # Limpia el estado interno del canvas para evitar superposiciones.
    def _reset_canvas_state(self):
        try:
            if getattr(self, "_last_draw_cid", None) is not None:
                self.canvas.mpl_disconnect(self._last_draw_cid)
                self._last_draw_cid = None
            self._ax_left = None
            self._ax_right = None
            self._im_left = None
            self._im_right = None
            self.figure.clear()
            self.canvas.draw_idle()
            QApplication.processEvents()
        except Exception:
            pass

    # -------------------------------
    # 3-3.- PROCESAMIENTO DE IMÁGENES
    # -------------------------------

    # Método principal para aplicar un mapa de color a la imagen seleccionada.
    def apply_colormap_menu(self):
        """
        - Valida que exista una imagen cargada.
        - Obtiene la versión en escala de grises.
        - Muestra un spinner mientras se ejecuta el worker en background.
        - Inicia el worker encargado de aplicar el colormap.
        """
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen.")
            return
        colormap_name = self.colormap_combo.currentText()
        if not colormap_name:
            self.show_error("Selecciona un mapa de color.")
            return

        # Mostrar spinner antes de iniciar el worker
        spinner_path = os.path.join(script_dir, "resources\\gui\\liquid_dot_loader.gif")
        self.show_inline_spinner(spinner_path)
        QApplication.processEvents()  # Forzar inicio de la animación del GIF

        # Limpieza segura del worker anterior (si existe)
        try:
            prev = getattr(self, "worker", None)
            if prev is not None:
                if not prev.isRunning():
                    try: prev.finished.disconnect()
                    except Exception: pass
                    try: prev.done.disconnect()
                    except Exception: pass
                    self.worker = None
                else:
                    try: prev.finished.disconnect()
                    except Exception: pass
                    try: prev.done.disconnect()
                    except Exception: pass
                    self.worker = None
        except Exception:
            self.worker = None

        # Crear nuevo worker para aplicar colormap
        self.worker = WorkerApplyColormap(imagen_gris, colormap_name)
        self.worker.finished.connect(self._on_worker_finished_update_canvas)
        self.worker.start()

    # Callback ejecutado cuando el worker termina de aplicar el colormap.
    def _on_worker_finished_update_canvas(self, datos):
        """
        - Actualiza el canvas con la imagen procesada.
        - Configura los ejes y títulos.
        - Oculta el spinner una vez que el render está completo.
        """
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return

        img_rgb, imagen_gris, colormap_name = datos
        if img_rgb is None:
            self.hide_inline_spinner()
            return

        h, w = img_rgb.shape[:2]
        recreate = False
        if not getattr(self, "_ax_left", None):
            recreate = True
        else:
            try:
                existing_shape = self._im_right.get_array().shape
                if existing_shape[:2] != (h, w):
                    recreate = True
            except Exception:
                recreate = True

        # Crear/recrear ejes si es necesario
        if recreate:
            self.figure.clear()
            axs = self.figure.subplots(1, 2)
            self._ax_left, self._ax_right = axs[0], axs[1]
            self._im_left = self._ax_left.imshow(np.zeros((h, w)), cmap='gray', vmin=0, vmax=255)
            self._im_right = self._ax_right.imshow(np.zeros((h, w, 3), dtype=np.uint8))
            self._ax_left.set_title('Imagen en escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
            self._ax_left.axis('off')
            self._ax_right.set_title(f'Pseudocolor: {colormap_name}', fontname=nombre_fuente, fontsize=tamano_fuente*2)
            self._ax_right.axis('off')
        else:
            self._ax_right.set_title(f'Pseudocolor: {colormap_name}', fontname=nombre_fuente, fontsize=tamano_fuente*2)

        # Actualizar datos en los ejes
        self._im_left.set_data(imagen_gris)
        self._im_left.set_clim(0, 255)
        self._ax_left.set_aspect('equal')

        self._im_right.set_data(img_rgb)
        self._ax_right.set_aspect('equal')

        # Ajustes de layout
        self.figure.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.05, wspace=0.02, hspace=0.05)

        # Ocultar spinner al terminar el dibujo
        def _on_draw(event):
            self.hide_inline_spinner()
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass

        cid = self.canvas.mpl_connect('draw_event', _on_draw)
        self.canvas.draw_idle()

        # Guardado diferido
        self.last_processed_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.last_action = "pseudocolor"

        if sender is self.worker:
            self.worker = None

    # Método principal para aplicar los mapa de color personalizados a la imagen seleccionada.
    def customize_colormap(self):
        """
        - Valida que haya imagen cargada.
        - Muestra loader mientras se ejecuta el worker.
        - Conecta señales para mostrar resultados al finalizar.
        """
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen.")
            return

        self.show_loader("Generando mapas personalizados...", "Procesando Imagen")
        self.worker = WorkerCustomizeColormap(imagen_gris)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.mostrar_personalizacion_mapas_gui)
        self.worker.finished.connect(self.progress_dialog.close)
        self.worker.start()

    # Callback ejecutado cuando el worker termina de aplicar los colormap personalizados.
    def mostrar_personalizacion_mapas_gui(self, imagen_gris):
        """
        Muestra en el canvas la comparación de varios colormaps personalizados.
        - Incluye mapas pastel, tierra y pastel personalizado.
        - Configura títulos y layout.
        """
        mapa_pastel = LinearSegmentedColormap.from_list("PastelMap", colores_pastel, N=256)
        mapa_tierra = LinearSegmentedColormap.from_list("TierraMap", colores_tierra, N=256)
        mapa_pastel_personalizado = LinearSegmentedColormap.from_list("PastelPersonalizadoMap", colores_pastel_personalizados, N=256)

        self.figure.clear()
        axs = self.figure.subplots(2, 2).flatten()

        axs[0].imshow(imagen_gris, cmap='gray')
        axs[0].set_title('Imagen en escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[0].axis('off')
        axs[0].set_aspect('equal')

        axs[1].imshow(imagen_gris, cmap=mapa_pastel)
        axs[1].set_title('Mapa de color pastel', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[1].axis('off')
        axs[1].set_aspect('equal')

        axs[2].imshow(imagen_gris, cmap=mapa_tierra)
        axs[2].set_title('Mapa de color tierra', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[2].axis('off')
        axs[2].set_aspect('equal')

        axs[3].imshow(imagen_gris, cmap=mapa_pastel_personalizado)
        axs[3].set_title('Mapa de color pastel personalizado', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[3].axis('off')
        axs[3].set_aspect('equal')

        # Maximizar uso de pantalla con márgenes mínimos
        self.figure.subplots_adjust(
            left=0.01,   # margen mínimo izquierdo
            right=0.99,  # margen mínimo derecho
            top=0.92,    # espacio justo para títulos
            bottom=0.05, # espacio inferior
            wspace=0.02, # separación horizontal mínima
            hspace=0.25  # separación vertical suficiente para títulos
        )

        # Render estable del canvas
        self.canvas.draw_idle()

        # Guardado diferido: no se guarda imagen procesada, solo se marca la acción
        self.last_processed_image = None
        self.last_action = "personalizacion"

    # Método agregado por iniciativa personal, para la comparación de los múltiples mapas de color disponibles en OpenCV.
    def compare_colormaps(self):
        """
        - Valida que haya una imagen cargada.
        - Convierte la imagen a escala de grises.
        - Muestra un loader con rango definido para indicar progreso real.
        - Crea y lanza un worker en background que realizará la comparación.
        """
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen.")
            return

        # Crear loader con rango definido para mostrar progreso real
        self.show_loader("Comparando mapas de color...", "Procesando Imágenes")

        # Crear worker para comparación
        self.worker = WorkerCompareColorMap(imagen_gris)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.mostrar_comparacion_mapas_gui)
        self.worker.start()

    # Callback ejecutado cuando el worker termina de aplicar los múltiples mapas de color.
    def mostrar_comparacion_mapas_gui(self, resultados):
        """
        Muestra en el canvas la comparación visual de varios mapas de color.
        - Organiza las imágenes en una cuadrícula.
        - Incluye la imagen original en escala de grises.
        - Actualiza el progreso paso a paso mientras dibuja.
        - Ajusta el layout y renderiza de forma estable.
        - Actualiza estado interno para guardado diferido.
        """
        total_imgs = len(resultados) + 1  # imagen original + resultados
        n_cols = min(5, total_imgs)       # máximo 5 columnas
        n_rows = int(np.ceil(total_imgs / n_cols))  # número de filas según total

        self.figure.clear()
        axs = self.figure.subplots(n_rows, n_cols).reshape(-1)

        # Imagen original en escala de grises
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        axs[0].imshow(imagen_gris, cmap='gray')
        axs[0].set_title('Escala de grises', fontname=nombre_fuente, fontsize=tamano_fuente*2)
        axs[0].axis('off')

        # Dibujar resultados con progreso incremental (30–95%)
        total = len(resultados)
        for idx, (nombre, imagen) in enumerate(resultados):
            axs[idx+1].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
            axs[idx+1].set_title(nombre, fontname=nombre_fuente, fontsize=tamano_fuente*2)
            axs[idx+1].axis('off')

            # Actualizar progreso paso a paso
            progreso = 20 + int((idx+1)/total * 75)  # hasta ~95%
            self.progress_dialog.setValue(progreso)
            QApplication.processEvents()

            # Dibujar en bloques intermedios para no congelar la UI
            if (idx+1) % n_cols == 0 and (idx+1) != total:
                self.canvas.draw_idle()
                QApplication.processEvents()

        # Desactivar ejes sobrantes (si hay más subplots que imágenes)
        for ax in axs[total_imgs:]:
            ax.axis('off')

        # Ajustar layout al final para que los títulos no se encimen
        self.figure.tight_layout(rect=[0, 0, 1, 1])

        # Render final síncrono
        self.canvas.draw_idle()
        QApplication.processEvents()

        # Finalizar loader justo después del render completo
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()

        # Guardado diferido: se marca la acción como comparación
        self.last_action = "comparacion"

# ------------------------------------------
# 4- PUNTO DE ENTRADA PRINCIPAL DEL PROGRAMA
# ------------------------------------------
if __name__ == "__main__":
    # Se crea una instancia de la aplicación Qt, pasando los argumentos del sistema. 
    # Esto inicializa el entorno gráfico necesario para ejecutar la GUI.
    app = QApplication(sys.argv)

    # Se establece la fuente global de la aplicación usando los valores definidos en config.py. 
    # Esto garantiza una apariencia consistente en todos los widgets.
    app.setFont(QFont(nombre_fuente, tamano_fuente))

    # Se crea una instancia de la clase principal de la GUI, y ae muestra la ventana principal en pantalla.
    viewer = Practica1GUI()
    viewer.show()

    # Se inicia el bucle de eventos de Qt. El programa se mantiene corriendo hasta que el usuario cierre la ventana.
    sys.exit(app.exec_())

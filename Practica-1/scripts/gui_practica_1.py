import os
import sys
import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QComboBox, QStyledItemDelegate, QMessageBox, 
    QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

# Módulos propios
from config import script_dir, mapas_color, colores_pastel, colores_tierra, colores_pastel_personalizados
from practica_1 import comparar_mapas_color
from imagen_pseudocolor import ImagenPseudocolor


class CenteredComboBoxDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class Practica1GUI(QMainWindow):

    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    def show_info(self, message: str):
        QMessageBox.information(self, "Información", message)

    def mostrar_pseudocolor(self, imagen_gris, colormap_name):
        try:
            resultado = ImagenPseudocolor.aplicar_pseudocolor(imagen_gris, colormap_name)
            self.figure.clear()
            axs = self.figure.subplots(1, 2)
            axs[0].imshow(imagen_gris, cmap='gray')
            axs[0].set_title('Imagen en escala de grises')
            axs[0].axis('off')
            axs[1].imshow(cv2.cvtColor(resultado.imagen, cv2.COLOR_BGR2RGB))
            axs[1].set_title(f'Pseudocolor: {colormap_name}')
            axs[1].axis('off')
            self.figure.tight_layout()
            self.canvas.draw()

            # Guardado diferido
            self.last_processed_image = resultado.imagen
            self.last_action = "pseudocolor"

            # Definir ruta para la figura actual (comparación)
            nombre_archivo_comp = f"comparacion_pseudocolor_{colormap_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
            os.makedirs(ruta_carpeta, exist_ok=True)
            self.ruta_imagen_comp = os.path.join(ruta_carpeta, nombre_archivo_comp)

        except ValueError as e:
            self.show_error(str(e))

    def mostrar_personalizacion_mapas_gui(self, imagen_gris):
        mapa_pastel = LinearSegmentedColormap.from_list("PastelMap", colores_pastel, N=256)
        mapa_tierra = LinearSegmentedColormap.from_list("TierraMap", colores_tierra, N=256)
        mapa_pastel_personalizado = LinearSegmentedColormap.from_list("PastelPersonalizadoMap", colores_pastel_personalizados, N=256)

        self.figure.clear()
        axs = self.figure.subplots(2, 2).flatten()
        axs[0].imshow(imagen_gris, cmap='gray')
        axs[0].set_title('Imagen en escala de grises')
        axs[0].axis('off')
        axs[1].imshow(imagen_gris, cmap=mapa_pastel)
        axs[1].set_title('Mapa de color pastel')
        axs[1].axis('off')
        axs[2].imshow(imagen_gris, cmap=mapa_tierra)
        axs[2].set_title('Mapa de color tierra')
        axs[2].axis('off')
        axs[3].imshow(imagen_gris, cmap=mapa_pastel_personalizado)
        axs[3].set_title('Mapa de color pastel personalizado')
        axs[3].axis('off')
        self.figure.tight_layout()
        self.canvas.draw()
        # Guardado diferido: figura matplotlib
        self.last_processed_image = None  # Guardamos la figura matplotlib
        self.last_action = "personalizacion" # <-- trackeamos acción

    def mostrar_comparacion_mapas_gui(self, imagen_gris):
        nombres_colormaps = list(mapas_color.keys())
        n_colormaps = len(nombres_colormaps)
        total_imgs = n_colormaps + 1  # +1 para la imagen en escala de grises

        n_cols = min(5, total_imgs)
        n_rows = int(np.ceil(total_imgs / n_cols))

        self.figure.clear()
        axs = self.figure.subplots(n_rows, n_cols).reshape(-1)

        # Imagen en escala de grises
        axs[0].imshow(imagen_gris, cmap='gray')
        axs[0].set_title('Escala de grises')
        axs[0].axis('off')

        # Pseudocolores
        for idx, nombre in enumerate(nombres_colormaps):
            pseudocolor = ImagenPseudocolor.aplicar_pseudocolor(imagen_gris, nombre)
            axs[idx+1].imshow(cv2.cvtColor(pseudocolor.imagen, cv2.COLOR_BGR2RGB))
            axs[idx+1].set_title(nombre)
            axs[idx+1].axis('off')

        # Ocultar ejes sobrantes
        for ax in axs[total_imgs:]:
            ax.axis('off')

        self.figure.tight_layout(rect=[0, 0, 1, 1])
        self.canvas.draw()

        # Guardado diferido: figura matplotlib
        self.last_processed_image = None
        self.last_action = "comparacion"


    def save_current_image(self):
        if self.last_action is None:
            self.show_error("Selecciona una opción válida antes de guardar la imagen.")
            return

        ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
        os.makedirs(ruta_carpeta, exist_ok=True)

        if self.last_action == "pseudocolor" and self.last_processed_image is not None:
            # Obtener colormap actual del combo
            colormap_name = self.colormap_combo.currentText().strip().replace(" ", "_")

            # Guardar imagen procesada (OpenCV)
            nombre_archivo = f"imagen_procesada_{colormap_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
            cv2.imwrite(ruta_imagen, self.last_processed_image)

            # Guardar la figura actual (comparación ya dibujada en el canvas)
            nombre_archivo_comp = f"comparacion_pseudocolor_{colormap_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ruta_imagen_comp = os.path.join(ruta_carpeta, nombre_archivo_comp)
            self.figure.savefig(ruta_imagen_comp, bbox_inches='tight', pad_inches=0.05)

            self.show_info(f"Imagen guardada en:\n{ruta_imagen}\nComparación guardada en:\n{ruta_imagen_comp}")

        elif self.last_action == "personalizacion":
            nombre_archivo = f"mapas_color_personalizados_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
            self.figure.savefig(ruta_imagen, bbox_inches='tight', pad_inches=0.05)
            self.show_info(f"Figura guardada en:\n{ruta_imagen}")

        elif self.last_action == "comparacion":
            nombre_archivo = f"comparacion_mapas_color_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
            self.figure.savefig(ruta_imagen, bbox_inches='tight', pad_inches=0.05)
            self.show_info(f"Comparación guardada en:\n{ruta_imagen}")


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Practica 1 - Menú Principal (GUI)")
        self.resize(1800, 1200)

        # Centrar ventana
        screen = QGuiApplication.primaryScreen().availableGeometry()
        center_point = screen.center()
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(center_point)
        self.move(frame_geom.topLeft())

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.btn_select_image = QPushButton("1. Seleccionar Imagen a Procesar")
        self.btn_select_image.clicked.connect(self.select_image)
        self.layout.addWidget(self.btn_select_image)

        self.btn_apply_colormap = QPushButton("2. Aplicar un Mapa de Color a la Imagen en Escala de Grises")
        self.btn_apply_colormap.clicked.connect(self.apply_colormap_menu)
        self.layout.addWidget(self.btn_apply_colormap)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(mapas_color.keys()))
        self.colormap_combo.setPlaceholderText("Selecciona un mapa de color")
        self.colormap_combo.setItemDelegate(CenteredComboBoxDelegate(self.colormap_combo))
        self.colormap_combo.setEditable(True)
        self.colormap_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.colormap_combo.lineEdit().setReadOnly(True)
        self.layout.addWidget(self.colormap_combo)

        self.btn_customize_colormap = QPushButton("3. Personalización del Mapa de Color")
        self.btn_customize_colormap.clicked.connect(self.customize_colormap)
        self.layout.addWidget(self.btn_customize_colormap)

        self.btn_compare_colormaps = QPushButton("4. Comparación Visual de Mapas de Color Disponibles en OpenCV")
        self.btn_compare_colormaps.clicked.connect(self.compare_colormaps)
        self.layout.addWidget(self.btn_compare_colormaps)

        self.btn_exit = QPushButton("5. Salir del Programa")
        self.btn_exit.clicked.connect(self.close)
        self.layout.addWidget(self.btn_exit)

        # Figura de matplotlib (canvas)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.canvas)

        self.image = None
        self.img_rgb = None
        self.imagen_path = None
        self.last_action = None # trackea la última opción seleccionada 
        self.last_processed_image = None # trackea la última imagen procesada (OpenCV) para guardado diferido

        # Espaciador flexible para empujar el botón hacia abajo
        # self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Botón para guardar la imagen procesada o la figura actual
        self.btn_save_image = QPushButton("Guardar Imagen Procesada")
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.layout.addWidget(self.btn_save_image)


    def select_image(self):
        initial_dir = os.path.join(script_dir, "resources", "input")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", initial_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.imagen_path = file_path
            self.image = cv2.imread(file_path)
            if self.image is None:
                self.show_error("Error al cargar la imagen. Asegúrate de seleccionar un archivo de imagen válido.")
                return
            self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.show_image()

    def show_image(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.img_rgb)
        ax.set_title("Imagen Original")
        ax.axis("off")
        self.canvas.draw()

    def apply_colormap_menu(self):
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada. Por favor, selecciona una imagen antes de aplicar un mapa de color.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen. Asegúrate de seleccionar un archivo de imagen válido.")
            return
        colormap_name = self.colormap_combo.currentText()
        if not colormap_name:
            self.show_error("Selecciona un mapa de color.")
            return
        self.mostrar_pseudocolor(imagen_gris, colormap_name)

    def customize_colormap(self):
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada. Por favor, selecciona una imagen antes de personalizar los mapas de color.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen. Asegúrate de seleccionar un archivo de imagen válido.")
            return
        self.mostrar_personalizacion_mapas_gui(imagen_gris)

    def compare_colormaps(self):
        if self.imagen_path is None:
            self.show_error("No hay imagen seleccionada. Por favor, selecciona una imagen antes de comparar los mapas de color.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            self.show_error("No se pudo cargar la imagen. Asegúrate de seleccionar un archivo de imagen válido.")
            return
        self.mostrar_comparacion_mapas_gui(imagen_gris)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = Practica1GUI()
    viewer.show()
    sys.exit(app.exec_())

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Practica1GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Practica 1 - Menú Principal (GUI)")
        self.setGeometry(100, 100, 900, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Botón para seleccionar imagen
        self.btn_select_image = QPushButton("1. Seleccionar Imagen a Procesar")
        self.btn_select_image.clicked.connect(self.select_image)
        self.layout.addWidget(self.btn_select_image)

        # Botón para aplicar mapa de color
        self.btn_apply_colormap = QPushButton("2. Aplicar un Mapa de Color a la Imagen en Escala de Grises")
        self.btn_apply_colormap.clicked.connect(self.apply_colormap_menu)
        self.layout.addWidget(self.btn_apply_colormap)

        # Botón para comparación de mapas
        self.btn_compare_colormaps = QPushButton("3. Comparación Visual de Mapas de Color Disponibles en OpenCV")
        self.btn_compare_colormaps.clicked.connect(self.compare_colormaps)
        self.layout.addWidget(self.btn_compare_colormaps)

        # Botón para personalización de mapas
        self.btn_customize_colormap = QPushButton("4. Personalización del Mapa de Color")
        self.btn_customize_colormap.clicked.connect(self.customize_colormap)
        self.layout.addWidget(self.btn_customize_colormap)

        # Botón para salir
        self.btn_exit = QPushButton("5. Salir del Programa")
        self.btn_exit.clicked.connect(self.close)
        self.layout.addWidget(self.btn_exit)

        # Figura de matplotlib
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.image = None
        self.img_rgb = None
        self.imagen_path = None

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.imagen_path = file_path
            self.image = cv2.imread(file_path)
            if self.image is None:
                print("Error al cargar la imagen.")
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
            print("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            print("No se pudo cargar la imagen.")
            return
        from practica_1 import menu_mapas_color
        menu_mapas_color(imagen_gris)

    def compare_colormaps(self):
        if self.imagen_path is None:
            print("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            print("No se pudo cargar la imagen.")
            return
        from practica_1 import comparar_mapas_color
        comparar_mapas_color(imagen_gris)

    def customize_colormap(self):
        if self.imagen_path is None:
            print("No hay imagen seleccionada.")
            return
        imagen_gris = cv2.imread(self.imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen_gris is None:
            print("No se pudo cargar la imagen.")
            return
        from practica_1 import mostrar_personalizacion_mapas
        mostrar_personalizacion_mapas(imagen_gris)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = Practica1GUI()
    viewer.show()
    sys.exit(app.exec_())

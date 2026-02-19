# Practica 1-a). "Creando mi mapa de calor"
import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

class ImagenPseudocolor:
    """
    Clase para almacenar el resultado de aplicar un mapa de color (pseudocolor) a una imagen en escala de grises.
        Atributos: 
        - nombre: str → identificador del mapa de color aplicado
        - imagen: np.ndarray → imagen pseudocoloreada
    """
    def __init__(self, nombre: str, imagen: np.ndarray) -> None:
        # nombre: str → asegura que el identificador del mapa de color sea una cadena
        self.nombre: str = nombre
        
        # imagen: np.ndarray → indica que la imagen pseudocolor es un arreglo NumPy
        # (OpenCV devuelve las imágenes en este formato)
        self.imagen: np.ndarray = imagen

    def mostrar(self) -> None:
        """
        Método auxiliar para visualizar la imagen pseudocolor con matplotlib.
        Usa el atributo 'nombre' como título de la figura.
        """
        plt.imshow(cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB))
        plt.title(f"Pseudocolor: {self.nombre}")
        plt.axis("off")
        plt.show()
    
    def guardar(self, ruta_base: str = "resultado") -> str:
        """
        Método auxiliar para guardar la imagen pseudocolor en disco.
        El archivo se nombra automáticamente con el colormap seleccionado.
        
        Parámetros:
            ruta_base (str): Nombre base del archivo (sin extensión).
        
        Retorna:
            str: Ruta completa del archivo guardado.
        """
        nombre_imagen = f"{ruta_base}_{self.nombre}.png"
        ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
        # Crear la carpeta si no existe
        os.makedirs(ruta_carpeta, exist_ok=True)
        ruta_imagen = os.path.join(ruta_carpeta, nombre_imagen)
        cv2.imwrite(ruta_imagen, self.imagen)
        return ruta_imagen

# Diccionario de mapas de color disponibles en OpenCV
# Obtener automáticamente todos los colormaps disponibles
mapas_color = {name.replace("COLORMAP_", ""): getattr(cv2, name) 
               for name in dir(cv2) if name.startswith("COLORMAP_")}

# ---- FUNCIONES DE PROCESAMIENTO DE IMAGENES ---------

# Función para aplicar un mapa de color (pseudocolor)
def aplicar_pseudocolor(imagen_gris, opcion):
    """
    Aplica un mapa de color (pseudocolor) determinado a una imagen en escala de grises usando OpenCV.
    Parámetros:
    - imagen_gris → imagen en escala de grises a la que se aplicará el pseudocolor
    - opcion: str → nombre del mapa de color a aplicar (ej. 'JET', 'HOT', 'OCEAN', etc.)
    Retorna: objeto ResultadoPseudocolor con la imagen pseudocoloreada
    """

    # Comparación insensible a mayúsculas/minúsculas
    opcion = opcion.upper()

    # Validar que la opción seleccionada esté en el diccionario de mapas de color
    if opcion not in mapas_color:
        raise ValueError(f"Opción '{opcion}' no válida. Opciones disponibles: {list(mapas_color.keys())}")
    
    # Aplicar el mapa de color seleccionado a la imagen en escala de grises usando OpenCV
    imagen_pseudocolor = ImagenPseudocolor(nombre=opcion, imagen=cv2.applyColorMap(imagen_gris, mapas_color[opcion]))
    
    return imagen_pseudocolor


# Función para comparar visualmente la imagen con varias versiones pseudocoloreadas
def comparar_mapas_color(imagen_gris):
    """
    Muestra la imagen en escala de grises y todas las versiones pseudocoloreadas disponibles.
    Organizando la cuadrícula de manera dinámica según la cantidad de paletas de OpenCV detectadas.
    """
    nombres_colormaps = list(mapas_color.keys())
    n_colormaps = len(nombres_colormaps)
    total_imgs = n_colormaps + 1  # +1 para la imagen en escala de grises

    # Calcular filas y columnas para la cuadrícula
    n_cols = min(5, total_imgs)  # máximo 5 columnas para mejor visualización
    n_rows = int(np.ceil(total_imgs / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axs = np.array(axs).reshape(-1)  # aplanar para indexar fácilmente

    # Imagen en escala de grises en la primera posición
    axs[0].imshow(imagen_gris, cmap='gray')
    axs[0].set_title('Escala de grises')
    axs[0].axis('off')

    # Mostrar cada pseudocolor
    for idx, nombre in enumerate(nombres_colormaps):
        pseudocolor = aplicar_pseudocolor(imagen_gris, nombre)
        axs[idx+1].imshow(cv2.cvtColor(pseudocolor.imagen, cv2.COLOR_BGR2RGB))
        axs[idx+1].set_title(nombre)
        axs[idx+1].axis('off')

    # Desactivar ejes sobrantes si hay
    for ax in axs[total_imgs:]:
        ax.axis('off')

    # Ajustar el layout para evitar recortes y superposiciones
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(bottom=0.03, top=0.97, left=0.03, right=0.97, hspace=0.25, wspace=0.15)
    # Guardar la figura antes de mostrarla, sin elementos extra
    nombre_archivo = f"comparacion_mapas_color_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    ruta_carpeta = os.path.join(script_dir, 'resources/pseudocolor')
    os.makedirs(ruta_carpeta, exist_ok=True)
    ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
    fig.savefig(ruta_imagen, bbox_inches='tight', pad_inches=0.05)
    print(f"Comparación guardada en: {ruta_imagen}")

    plt.show()


# Obtener el directorio del script y regresar un nivel en la jerarquía de carpetas
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- FUNCIONES DE MENÚ ----------------
def menu_mapas_color(imagen_gris):
    while True:
        print("\n=== Menú de Mapas de Color Disponibles ===")
        for i, nombre in enumerate(mapas_color.keys(), start=1):
            print(f"{i}. {nombre}")
        print(f"{len(mapas_color)+1}. Regresar al Menú Principal [Regresar, Salir]")

        opcion_usuario = input("Selecciona un mapa de color por nombre o número: ").strip()

        # Permitir selección por número además de por nombre
        if opcion_usuario.isdigit():
            indice = int(opcion_usuario) - 1
            if indice == len(mapas_color):
                print("Regresando al Menú Principal...")
                break
            elif 0 <= indice < len(mapas_color):
                opcion_usuario = list(mapas_color.keys())[indice]
            else:
                print("Número fuera de rango.")
                continue
        elif opcion_usuario.upper() in ["REGRESAR", "SALIR"]:
            print("Regresando al Menú Principal...")
            break

        try:
            resultado = aplicar_pseudocolor(imagen_gris, opcion_usuario)
            resultado.mostrar()
            image_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ruta_guardada = resultado.guardar(image_name)
            print(f"Imagen guardada en: {ruta_guardada}")
            pausa = input("Presiona Enter para continuar...")
        except ValueError as e:
            print("Error:", e)


def menu_principal():
    # print("Directorio del script:", script_dir)

    # Se obtine la ruta completa de la imagen
    imagen_path = os.path.join(script_dir, 'resources/rostro-humano.jpg')
    print("Ruta de la imagen:", imagen_path)

    # Cargar imagen en escala de grises
    imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    if imagen_gris is None:
        raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta y extensión.")

    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Aplicar un Mapa de Color a la Imagen en Escala de Grises")
        print("2. Comparación Visual de Mapas de Color Disponibles en OpenCV")
        print("3. Salir del Programa")
        opcion = input("Selecciona una Opción: ").strip()

        if opcion == "1":
            menu_mapas_color(imagen_gris)
        elif opcion == "2":
            comparar_mapas_color(imagen_gris)
        elif opcion == "3" or opcion.upper() == "SALIR":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    menu_principal()

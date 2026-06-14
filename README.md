# Geometría Epipolar

Proyecto de modelización matemática centrado en la visión estereoscópica, la geometría de dos vistas y la reconstrucción tridimensional. El repositorio contiene tanto la fundamentación teórica (memoria) como la implementación práctica para la obtención de mapas de profundidad.

## Autores
* Álvaro Acedo Blanco
* Daniel Czepiel Babiarz
* Seán Beltrán Amor

## Estructura del Proyecto

El repositorio está dividido en dos bloques principales:

* **Teoria/**: Contiene el código fuente en LaTeX (`geometria_epipolar.tex`), los diagramas (`Figures/`), las referencias bibliográficas (`sample.bib`) y el documento compilado final (`geometria_epipolar.pdf`). Este documento detalla la transición desde el modelo de cámara estenopeica ideal hasta la cámara finita general, la definición de las matrices fundamental y esencial, y la formulación para la rectificación de imágenes y cálculo de disparidad.
* **Practica/**: Contiene la implementación en Python.
  * `main.py`: Script principal que carga un par estéreo rectificado (`im0.png`, `im1.png`) y utiliza el algoritmo Semi-Global Block Matching (SGBM) de OpenCV para calcular el mapa de disparidad. Exporta los resultados en escala de grises y con mapa de color.
  * `calculateF.py`: Script orientado al cálculo de la matriz fundamental y el procesamiento analítico entre correspondencias.
  * Resultados: `mapa_disparidad_gris.png` y `mapa_disparidad_color.png`.
* **Multiple View Geometry in Computer Vision...pdf**: Libro de referencia principal utilizado para el desarrollo algebraico.

## Requisitos y Configuración

El código de la carpeta `Practica/` requiere Python 3 y las siguientes librerías:
* `numpy`
* `opencv-python` (cv2)

Para ejecutar el código, se recomienda activar el entorno virtual incluido o instalar las dependencias y ejecutar el script principal:

```bash
# Activar entorno virtual (Linux/macOS)
source venv/bin/activate

# Ejecutar la generación del mapa de disparidad
cd Practica
python main.py
```

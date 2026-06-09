"""
main.py — Pipeline estéreo automático
Uso: python main.py im0.png im1.png
"""

import sys
import numpy as np
import cv2
from calculos import stereo_depth


# ============================================================
# DETECCIÓN AUTOMÁTICA DE CORRESPONDENCIAS (SIFT)
# ============================================================

def find_correspondences(img_left, img_right, max_features=2000):
    """
    Detecta correspondencias entre las dos imágenes usando SIFT + ratio test.
    Devuelve pts1, pts2 arrays (N, 2) listos para pasar a stereo_depth.

    Sift detecta keypoints y descriptores:
    Keypoints: Guardan la localización exacta del punto en la imagen.
    Descriptores: Describen la textura del entorno de cada keypoint,
    permitiendo comparar esta descripción con la de un punto de otra
    imagen y valorar si se trata del mismo punto físico
    """
    sift = cv2.SIFT_create(max_features)
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    #El algoritmo utilizado para calcular F necesita al menos 8 puntos
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("No se encontraron suficientes keypoints SIFT.")

    # Para cada descriptor de una imagen, se buscan los dos más parecidos
    # en la otra
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test de Lowe — filtra matches ambiguos
    # Solo si el más parecido es claramente mejor que el segundo se coje,
    # en otro caso se filtra
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        raise RuntimeError(f"Solo {len(good)} matches tras el ratio test. Necesita al menos 8.")

    # Representan las coordenadas en cada imagen de las correspondencias
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    print(f"  Correspondencias SIFT encontradas: {len(good)}")
    return pts1, pts2


# ============================================================
# VISUALIZACIÓN
# ============================================================

def normalize_for_display(arr):
    """Normaliza un array float a uint8 [0, 255] usando todo el rango."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def normalize_depth_for_display(depth):
    """
    Normaliza el mapa de profundidad a uint8 ignorando los píxeles
    inválidos (depth == 0), que calculos.py marca así cuando la
    disparidad era menor que min_disp.

    Sin esto, el valor 0 (sin datos) entra en el rango de normalización
    como si fuera la profundidad mínima, aplastando todo el detalle
    del resto de la imagen.
    """
    out   = np.zeros_like(depth, dtype=np.uint8)
    valid = depth > 0
    if valid.sum() == 0:
        return out
    mn, mx = depth[valid].min(), depth[valid].max()
    if mx - mn < 1e-6:
        out[valid] = 128
        return out
    out[valid] = ((depth[valid] - mn) / (mx - mn) * 255).astype(np.uint8)
    return out


def colormap(gray_uint8):
    """Aplica colormap INFERNO para mejor percepción de profundidad."""
    return cv2.applyColorMap(gray_uint8, cv2.COLORMAP_INFERNO)


def show_results(imgL, imgR, imgLr, imgRr, disp, depth):
    """Muestra un panel con los 6 resultados del pipeline."""
    h, w = imgL.shape

    # Convertir a BGR para poder mezclar con imágenes de color
    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    disp_vis  = colormap(normalize_for_display(disp))
    depth_vis = colormap(normalize_depth_for_display(depth))

    row1 = np.hstack([to_bgr(imgL),  to_bgr(imgR)])
    row2 = np.hstack([to_bgr(imgLr), to_bgr(imgRr)])
    row3 = np.hstack([disp_vis,      depth_vis])

    panel = np.vstack([row1, row2, row3])

    # Etiquetas
    font    = cv2.FONT_HERSHEY_SIMPLEX
    labels  = [
        (10, 30,  "Original izquierda"),
        (w + 10, 30,  "Original derecha"),
        (10, h + 30,  "Rectificada izquierda"),
        (w + 10, h + 30, "Rectificada derecha"),
        (10, 2*h + 30,   "Disparidad"),
        (w + 10, 2*h + 30, "Profundidad relativa"),
    ]
    for x, y, text in labels:
        cv2.putText(panel, text, (x, y), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Pipeline estéreo", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main() :
    # Suponemos que las imágenes a cargar tienen estos nombres si
    # no se han pasado sus nombres como parámetros
    left_path  = sys.argv[1] if len(sys.argv) > 1 else "im0.png"
    right_path = sys.argv[2] if len(sys.argv) > 2 else "im1.png"

    # Cargamos las imágenes en escala de grises para simplificar el
    # problema, pues la visión estereo no necesita color para calcular 
    # la geometría
    print(f"Cargando imágenes: {left_path}, {right_path}")
    imgL = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Error: no se pudieron cargar las imágenes.")
        sys.exit(1)

    # Obtención de puntos que representan el mismo punto físico en
    # ambas imágenes
    print("Detectando correspondencias SIFT...")
    pts1, pts2 = find_correspondences(imgL, imgR)

    print("Ejecutando pipeline estéreo...")
    # Delegamos todos los cálculos a esta función
    # Pasamos las imágenes y las correspondencias
    # Lo demás son valores por defecto que dan buenos resultados
    disp, depth, imgLr, imgRr, F, mask = stereo_depth(
        imgL, imgR, pts1, pts2,
    )

    print(f"  Inliers RANSAC: {mask.sum()} / {len(mask)}")
    print(f"  Matriz fundamental F:\n{F}")

    print("Guardando resultados...")
    cv2.imwrite("output/rect_left.png",   imgLr)
    cv2.imwrite("output/rect_right.png",  imgRr)
    cv2.imwrite("output/disparity.png",       normalize_for_display(disp))
    cv2.imwrite("output/depth.png",           normalize_depth_for_display(depth))
    cv2.imwrite("output/disparity_color.png", colormap(normalize_for_display(disp)))
    cv2.imwrite("output/depth_color.png",     colormap(normalize_depth_for_display(depth)))

    print("Mostrando resultados (pulsa cualquier tecla para cerrar)...")
    show_results(imgL, imgR, imgLr, imgRr, disp, depth)


if __name__ == "__main__":
    main()

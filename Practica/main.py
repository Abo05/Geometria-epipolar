import cv2
import numpy as np

def main():
    # 1. Cargar las imágenes (Middlebury suele tener 'im0.png' e 'im1.png')
    # Es fundamental cargarlas en escala de grises para el cálculo de disparidad
    img_left_path = 'im0.png'   # Cambia esto por la ruta de tu imagen izquierda
    img_right_path = 'im1.png'  # Cambia esto por la ruta de tu imagen derecha

    imgL = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Error: No se pudieron cargar las imágenes. Verifica las rutas.")
        return

    # 2. Configurar los parámetros de StereoSGBM
    # Estos parámetros están pre-ajustados para funcionar bien con datasets tipo Middlebury
    window_size = 5  # Tamaño del bloque (debe ser impar: 3, 5, 7...)
    min_disp = 0
    num_disp = 16 * 8  # Debe ser divisible por 16. Aumentar si las cámaras están muy separadas.

    stereo = cv2.StereoSGBM_create( # type: ignore
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,   # Penalización por cambios suaves de disparidad
        P2=32 * 3 * window_size**2,  # Penalización por saltos bruscos de disparidad
        disp12MaxDiff=1,
        uniquenessRatio=10,          # Margen de victoria del mejor *match* (en %)
        speckleWindowSize=100,       # Tamaño de regiones de ruido a eliminar
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )

    print("Calculando mapa de disparidad con OpenCV...")
    
    # 3. Calcular la disparidad
    # OpenCV devuelve la disparidad multiplicada por 16 (formato de punto fijo de 12 bits)
    disparity_16sgbm = stereo.compute(imgL, imgR)

    # Convertir a float32 y dividir entre 16 para obtener la disparidad real en píxeles
    disparity = disparity_16sgbm.astype(np.float32) / 16.0

# 4. Normalización para visualización
    # Creamos la matriz de destino vacía
    disp_vis = np.zeros(disparity.shape, dtype=np.uint8)

    # Normalizamos pasando 'disp_vis' como segundo argumento (sin hacer 'disp_vis = ...')
    cv2.normalize(
        src=disparity,
        dst=disp_vis,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    # (Opcional) Aplicar un mapa de color para que los relieves se aprecien mejor
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # 5. Guardar y mostrar los resultados
    cv2.imwrite('mapa_disparidad_gris.png', disp_vis)
    cv2.imwrite('mapa_disparidad_color.png', disp_color)
    print("Mapas de disparidad guardados con éxito.")

    # Mostrar en ventana (presiona cualquier tecla para cerrar)
    cv2.imshow('Imagen Izquierda', imgL)
    cv2.imshow('Mapa de Disparidad (Color)', disp_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

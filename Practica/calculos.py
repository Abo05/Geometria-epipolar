"""
Pipeline de visión estéreo sin dependencias de OpenCV
excepto para la detección de correspondencias SIFT y rectificación de las
imagen para la obtención de rectas epipolares horizontales
"""

import numpy as np
from numpy.linalg import svd
import cv2


# ============================================================
# 1. FUNDAMENTAL MATRIX — algoritmo 8 puntos normalizado + RANSAC
# ============================================================

def _compute_normalization_matrix(pts):
    """
    Mejora de Hartley para la estabilidad numérica del algoritmo de los
    8 puntos: normaliza los puntos para que tengan centroide en el origen
    y distancia RMS al origen igual a sqrt(2).
 
    Sin esta normalización, las coordenadas de imagen (valores de cientos
    a miles de píxeles) crean una matriz A muy mal condicionada y la SVD
    produce una F con errores numéricos grandes.
    """
    
    centroid = pts.mean(axis=0)
    pts_c    = pts - centroid
    rms      = np.sqrt((pts_c ** 2).sum(axis=1).mean())
    scale    = np.sqrt(2) / (rms + 1e-10)
    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,         0,                    1],
    ])
    return T


def _normalize(pts, T):
    """ Aplica la transformación homogénea T a pts"""
    N = pts.shape[0]
    h = np.hstack([pts, np.ones((N, 1))])
    return (T @ h.T).T[:, :2]


def _build_A(pts1n, pts2n):
    """
    Matriz de diseño A para el sistema A·f = 0, donde f son los
    9 elementos de F. Cada correspondencia x x' aporta una fila
    """
    x,  y  = pts1n[:, 0], pts1n[:, 1]
    xp, yp = pts2n[:, 0], pts2n[:, 1]
    ones   = np.ones(len(x))
    return np.column_stack([
        xp*x, xp*y, xp,
        yp*x, yp*y, yp,
        x,    y,    ones,
    ])


def _fundamental_8point(pts1, pts2):
    """
    Algoritmo de los 8 puntos normalizado creado por Hartley
 
    Pasos:
      1. Normalizar puntos (mejora la estabilidad numérica).
      2. Construir A y resolver A·f=0 con SVD: la solución es el
         vector singular derecho de menor valor singular (última fila de Vt).
      3. Imponer rango 2 (poner a cero el menor valor singular de F).
      4. Desnormalizar: F_px = T2^T · F_norm · T1.
    """
    T1  = _compute_normalization_matrix(pts1)
    T2 = _compute_normalization_matrix(pts2)

    pts1n = _normalize(pts1, T1)
    pts2n = _normalize(pts2, T2)

    # Cada correspondencia genera una ecuación, sea A un sistema que
    # contiene estas ecuaciones
    A = _build_A(pts1n, pts2n)

    # Resolución del sistema factorizando la matriz A en tres matrices 
    # simples para facilitar la solución de sistemas de ecuaciones con
    # más ecuaciones que incógnitas
    _, _, Vt = np.linalg.svd(A)
    
    # La solución dada en Vt[-1] es un vector de 9 elementos que lo
    # convertimos en la matriz 3x3 que queremos
    F_hat = Vt[-1].reshape(3, 3)

    # Obligamos que la matriz fundamental tenga rango 2
    U, S, Vt2 = np.linalg.svd(F_hat)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt2
    F = T2.T @ F_rank2 @ T1

    # Renormalizar por la norma de Frobenius
    fnorm = np.linalg.norm(F)
    if fnorm > 1e-10:
        F /= fnorm
    return F


def _epipolar_residuals(F, pts1, pts2):
    """
    Distancia de cada punto x' a su línea epipolar l'=F·x (en píxeles).
    Mide cuánto se aleja cada correspondencia de la restricción epipolar x'^T·F·x=0.
    Un buen F da residuos menores que 1-2 px.
    """
    N    = len(pts1)
    # Convertimos los puntos a coordenadas homogéneas
    p1h  = np.hstack([pts1, np.ones((N, 1))])
    p2h  = np.hstack([pts2, np.ones((N, 1))])

    # Calculamos las líneas epipolares
    lines = (F @ p1h.T).T

    # Distancia perpendicular del punto a la línea epipolar utilizando
    # la fórmula de distancia entre un punto y una recta.
    # El resultado está expresado en píxeles.
    num   = np.abs((p2h * lines).sum(axis=1))
    den   = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2) + 1e-10
    return num / den


def compute_fundamental(pts1, pts2,
                        threshold_px=0.5, iterations=5000):
    """
    Implementación del algoritmo RANSAC para estimar la matriz fundamental
    """
    # Pasamos de float32 a float64 para mejorar la estabilidad numérica
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)

    N    = len(pts1)
    assert N >= 8, "Se necesitan al menos 8 correspondencias."

    # Generador aleatorio con semilla fija para obtener resultados reproducibles.
    rng        = np.random.default_rng(42)

    best_F     = None
    best_mask  = np.zeros(N, dtype=bool)
    best_count = 0

    # Bucle principal del RANSAC
    for _ in range(iterations):
        # Escogemos 8 correspondencias aleatorios
        idx = rng.choice(N, 8, replace=False)
        try:
            # Intentamos calcular la matriz F
            F_cand = _fundamental_8point(pts1[idx], pts2[idx])
        except Exception:
            continue
        # Calculamos su residuo epipolar, esto es, la distancia 
        # entre el punto detectado en la imagen y su línea epipolar.
        # res un array de residuos, uno para cada correspondencia
        res     = _epipolar_residuals(F_cand, pts1, pts2)

        # Guardamos en inliers aquellos errores que aceptamos, es decir,
        # cuyo error epipolar está por debajo del umbral establecido.
        # Si su cantidad de puntos cuyos errores aceptamos es mayor
        # que el de la mejor F del momento, la cambiamos
        inliers = res < threshold_px
        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_mask  = inliers
            best_F     = F_cand

    # Una vez terminado RANSAC, disponemos de un conjunto de inliers(best_mask).
    # Recalculamos F usando TODOS ellos para obtener una estimación
    # más precisa que la obtenida con solo 8 puntos.
    if best_mask.sum() >= 8:
        # Volvemos a calcular la matriz fundamental, pero ahora solo con
        # aquellos puntos buenos (inliers)
        best_F = _fundamental_8point(pts1[best_mask], pts2[best_mask])

        # Normalizamos F para evitar escalas arbitrarias
        fnorm = np.linalg.norm(best_F)
        if fnorm > 1e-10:
            best_F /= fnorm

    return best_F, best_mask


# ===========
# 2. EPIPOLES
# ===========

def epipole(F):
    """
    Calcula el epipolo como el vector e tal que F·e=0. 
    Devuelto normalizado, salvo si está en el infinito, 
    que es consecuencia de camaras paralelas
    """
    _, _, Vt = svd(F)
    e = Vt[-1]
    if abs(e[2]) > 1e-10:
        return e / e[2]
    return e   # epipolo en el infinito (cámaras paralelas)


# ================
# 3. RECTIFICACIÓN
# ================

def compute_rectification(F, img_shape, pts1_inliers, pts2_inliers):
    """
    Calcula las homografías de rectificación H1 y H2.

    La implementación utiliza stereoRectifyUncalibrated de OpenCV,
    que implementa una variante robusta del método de rectificación
    no calibrada propuesto por Hartley.

    El objetivo de la rectificación es transformar ambas imágenes
    de forma que las líneas epipolares se vuelvan horizontales y
    coincidan entre sí. Tras este proceso, la búsqueda de
    correspondencias se reduce a una búsqueda unidimensional a lo
    largo de las filas de la imagen.

    Conceptualmente, una rectificación proyectiva puede construirse:

        1. Calculando los epípolos e y e'.
        2. Aplicando una homografía que envíe cada epípolo al infinito.

            1. T — traslada el centro de la imagen al origen.
         
            2. R — rota para que el epipolo, ahora expresado relativo al centro,
               quede alineado con el eje X positivo: e_rotado = (f, 0, 1).
         
            3. G — transformación proyectiva que manda (f,0,1) al infinito (f,0,0):
                   G = [[1, 0, 0],
                        [0, 1, 0],
                        [-1/f, 0, 1]]
         
            H = G · R · T
        3. Consiguiendo así que todas las líneas epipolares sean paralelas.
        4. Aplicando transformaciones adicionales para minimizar
           distorsiones geométricas.

    Se ha optado por utilizar opencv en vez de implementar 1 y 2 para 
    la minimización de distorsiones geométricas
    """
    h, w = img_shape[:2]

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1_inliers.astype(np.float32),
        pts2_inliers.astype(np.float32),
        F,
        imgSize=(w, h),
        threshold=5.0,
    )
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated falló")

    return H1.astype(np.float64), H2.astype(np.float64)


# ===============================================
# 4. REMAPEO DE LA IMAGEN MEDIANTE UNA HOMOGRAFÍA
# ===============================================

def warp_image(img, H):
    """ 
    Aplica una homografía a la imagen mediante remapeo inverso.

    En nuestro caso, la homografía H ha sido calculada durante la
    rectificación y transforma la imagen de manera que las líneas
    epipolares pasan a ser horizontales. Tras esta transformación,
    la búsqueda de correspondencias se reduce a una única dimensión:
    para un píxel dado solo es necesario buscar sobre la misma fila
    de la imagen rectificada.
    """
    # Obtenemos las dimensiones de la imagen
    h, w   = img.shape[:2]

    # Invertimos la homografía.
    # Utilizamos remapeo invero. Para cada pixel de la imagen de salida
    # calculamos de qué posición de la imagen original procede.
    # Este enfoque evita huecos y regiones sin información que aparacerían
    # al proyectar directamente los píxeles de entrada.
    H_inv  = np.linalg.inv(H)

    # Generamos una rejilla con todas las coordenadas de la imagen rectificada
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    ones   = np.ones_like(xs)

    # Convertimos los píxeles a coordenadas homogeneas y los agrupamos
    # en una sola matriz para realizar operaciones de forma vectorizada
    pts    = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T

    # Aplicamos la homografía inversa para obtener la posición correspondiente
    # en la imagen original
    src    = H_inv @ pts

    # Volvemos a coordenadas cartesianas
    src   /= src[2]

    # Redondeamos cada coordenada al píxel más cercano
    x_src  = np.round(src[0]).astype(np.int32).reshape(h, w)
    y_src  = np.round(src[1]).astype(np.int32).reshape(h, w)

    # Mantenemos solo aquellos píxeles que se encuentran dentro de la imagen
    valid  = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)

    # Ponemos de fondo una imagen en negro y encima la imagen resultado
    out           = np.zeros_like(img)
    out[valid]    = img[y_src[valid], x_src[valid]]

    return out


# ============================================================
# 5. CÁLCULO DEL MAPA DE DISPARIDAD
# ============================================================

def box_sum(arr, block_size):
    """
    Calcula la suma de cada ventana block_size × block_size
    mediante imágenes integrales (sumas acumuladas).

    Gracias a ello, el coste de cada ventana puede obtenerse
    en tiempo constante independientemente de su tamaño.
    """
    # Acumulación vertical
    s = np.cumsum(arr, axis=0)
    s = s[block_size:] - s[:-block_size]

    # Acumulación horizontal
    s = np.cumsum(s, axis=1)
    s = s[:, block_size:] - s[:, :-block_size]
    return s

def compute_disparity(img0, img1, max_disp=64, block_size=9,
                      uniqueness=0.1, median_ksize=5):
    """
    Calcula el mapa de disparidad mediante block matching.
 
    Asume imágenes rectificadas: la correspondencia de un píxel (x,y)
    de img1 solo puede estar en la fila y de img2, desplazada d píxeles
    a la izquierda (d ∈ [0, max_disp)).
 
    Para cada disparidad candidata d se compara una ventana
    block_size × block_size utilizando la métrica SSD
    (Sum of Squared Differences):

        SSD = sum ((I1 - I2)^2)

    La disparidad asignada a cada píxel es aquella que minimiza
    dicho coste.

    La dirección del desplazamiento:
    La cámara derecha está a la derecha de la izquierda, así que el mismo
    objeto aparece más a la IZQUIERDA en img1. Para alinear img1 con img0
    se desplaza img1 d píxeles a la DERECHA: shifted[:, d:] = img1[:, :w-d].
    Esto es equivalente a comparar img0[x] con img1[x-d] para cada x. 

    """
    # Dimensiones de las imágenes rectificadas
    h, w  = img0.shape[:2]

    # Conversion de coma flotante para evitar problemas de precisión
    i1    = img0.astype(np.float32)
    i2    = img1.astype(np.float32)

    # Radio de la ventana
    pad   = block_size // 2
 
    # Guardamos los dos mejores costes para el test de unicidad
    best_cost        = np.full((h, w), np.inf, dtype=np.float32)
    second_best_cost = np.full((h, w), np.inf, dtype=np.float32)

    disp_map         = np.zeros((h, w), dtype=np.float32)
 
    # Recorremos todas las disparidades posibles
    # Si un objeto está más cerca de la cámara, tendremos que desplazar
    # más la cámara a la derecha. La disparidad representa este desplazamiento
    for d in range(max_disp):
        shifted        = np.zeros_like(i2)

        if d > 0:
            shifted[:, d:] = i2[:, :w - d]
        else:
            shifted = i2
 
        # Calculamos el coste acumulado en una ventana sobre la diferencia
        # cuadrática entre ambas imágenes
        cost = box_sum((i1 - shifted) ** 2, block_size)
 
        # box_sum reduce el tamaño de la imagen debido al borde. Recuperamos 
        # las dimensiones originales replicando los valores de los extremos.
        dh = h - cost.shape[0]
        dw = w - cost.shape[1]

        cost_full = np.pad(cost,
                           ((pad, dh - pad), (pad, dw - pad)),
                           mode='edge')
 
        # Comprobamos si esta disparidad proporciona un coste mejor que 
        # los observados hasta ahora.        
        is_best   = cost_full < best_cost
        is_second = (~is_best) & (cost_full < second_best_cost)

        # El anterior mejor pasa a segundo
        second_best_cost[is_best]   = best_cost[is_best]   
        second_best_cost[is_second] = cost_full[is_second]
 
        best_cost[is_best] = cost_full[is_best]

        disp_map[is_best]  = d
 
    # Si el mejor coste y el segundo mejor son demasiado parecidos,
    # significa que existen varias correspondencias igualmente
    # plausibles y el emparejamiento es ambiguo.
    #
    # Estos casos aparecen frecuentemente en regiones uniformes,
    # superficies repetitivas o zonas con poco contraste.
    #
    # Los descartamos asignándoles disparidad cero.
    if uniqueness > 0:
        ambiguous          = best_cost * (1.0 + uniqueness) > second_best_cost
        disp_map[ambiguous] = 0
 
    # Eliminamos valores aislados erróneos, ruido "sal y pimienta", difuminandolos
    if median_ksize > 1:
        disp_map = cv2.medianBlur(disp_map, median_ksize)
        disp_map = cv2.medianBlur(disp_map, median_ksize)
        
    return disp_map


# ============================================================
# 6. CONVERSIÓN DE DISPARIDAD A PROFUNDIDAD RELATIVA
# ============================================================

def disparity_to_depth(disp, baseline=1.0, focal=1.0, min_disp=0.5):
    """
    Convierte un mapa de disparidad en un mapa de profundidad gracias 
    a la relación geométrica fundamental en estéreo rectificado:
        Z = (focal * baseline) / disparity
        
    donde:
        Z         -> profundidad
        focal     -> distancia focal de la cámara
        baseline  -> distancia entre cámaras
        disparity -> desplazamiento horizontal entre correspondencias

    Nota importante:
    La profundidad aquí es relativa si focal y baseline no están
    calibrados en unidades métricas reales.
    """
    depth = np.zeros_like(disp, dtype=np.float32)

    # Evitamos disparidades nulas
    valid = disp >= min_disp

    depth[valid] = (focal * baseline) / disp[valid]

    # Normalizamos al rango [0,1] e invertimos para que cercano=brillante
    # y sea más visual
    valid_vals = depth[valid]
    if len(valid_vals) > 0:
        d_min, d_max = valid_vals.min(), valid_vals.max()

        # Evita la división por cero si la escena es plana
        if d_max > d_min:
            depth[valid] = 1.0 - (depth[valid] - d_min) / (d_max - d_min)

    return depth

# ============================================================
# 7. PIPELINE COMPLETO
# ============================================================

def stereo_depth(imgL, imgR, pts1, pts2,
                 use_ransac=True, threshold_px=1.5, ransac_iters=1000,
                 max_disp=64, block_size=15):
    """
    Pipeline completo de visión estéreo:

        1. Estimación de la geometría epipolar (F)
        2. Rectificación de imágenes
        3. Cálculo de disparidad (block matching)
        4. Conversión a profundidad relativa

    Devuelve:
        disp   -> mapa de disparidad
        depth  -> mapa de profundidad normalizado
        imgLr  -> imagen izquierda rectificada
        imgRr  -> imagen derecha rectificada
        F      -> matriz fundamental estimada
        mask   -> inliers de RANSAC
    """
    # Calculamos la matriz fundamental F a partir de las correspondencias
    # y usamos RANSAC para eliminar outliers
    F, mask = compute_fundamental(
        pts1, pts2,
        threshold_px=threshold_px,
        iterations=ransac_iters,
    )

    # A partir de F y de los inliers se calculan dos homografías
    # H1 y H2 que transforman las imágenes de forma que las líneas epipolares
    # se vuelven horizontales. Por lo tanto, la busqueda de correspondencias
    # se reduce a una dimensión
    H1, H2 = compute_rectification(F, imgL.shape, pts1[mask], pts2[mask])

    # Aplicamos las homografías
    imgLr = warp_image(imgL, H1)
    imgRr = warp_image(imgR, H2)

    # Con las imágenes rectificadas, reducimos el problema a buscar correspondencias
    # en una misma fila
    disp  = compute_disparity(imgLr, imgRr, max_disp=max_disp, block_size=block_size)

    depth = disparity_to_depth(disp)

    return disp, depth, imgLr, imgRr, F, mask

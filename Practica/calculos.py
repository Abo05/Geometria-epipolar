"""
calculos.py — Pipeline de visión estéreo sin dependencias de OpenCV
excepto para la detección de correspondencias SIFT.
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
    y distancia RMS al origen igual a √2.
 
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
      3. Imponer rango 2 zeroing el menor valor singular de F.
      4. Desnormalizar: F_px = Tp^T · F_norm · T.
    """
    T  = _compute_normalization_matrix(pts1)
    Tp = _compute_normalization_matrix(pts2)

    pts1n = _normalize(pts1, T)
    pts2n = _normalize(pts2, Tp)

    # Cada correspondencia genera una ecuación, sea A un sistema que
    # contiene estas ecuaciones
    A = _build_A(pts1n, pts2n)

    # Factorización de la matriz A en tres matrices simples para
    # facilitar la solución de sistemas de ecuaciones con más ecuaciones
    # que incógnitas
    _, _, Vt = np.linalg.svd(A)
    
    # La solución dada en Vt[-1] es un vector de 9 elementos que lo
    # convertimos en la matriz 3x3 que queremos
    F_hat = Vt[-1].reshape(3, 3)

    # Obligamos que la matriz fundamental tenga rango 2
    U, S, Vt2 = np.linalg.svd(F_hat)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt2
    F = Tp.T @ F_rank2 @ T

    # Normalizamos la matriz de tal forma que el último elemento sea 1
    if abs(F[2, 2]) > 1e-10:
        F /= F[2, 2]
    # Caso especial donde el elemento F[2,2] se acerca a 0
    # Por ejemplo, cuando el único movimiento que hacemos es mover la
    # cámara hacia delante, manteniendo el mismo origen de coordenadas
    # Esto significa que el epipolo también está en el centro de coordenadas
    return F


def _epipolar_residuals(F, pts1, pts2):
    """
    Distancia de cada punto x' a su línea epipolar l'=F·x (en píxeles).
    Mide cuánto viola cada correspondencia la restricción epipolar x'^T·F·x=0.
    Un buen F da residuos menores que 1-2 px.
    """
    N    = len(pts1)
    p1h  = np.hstack([pts1, np.ones((N, 1))])
    p2h  = np.hstack([pts2, np.ones((N, 1))])
    lines = (F @ p1h.T).T
    num   = np.abs((p2h * lines).sum(axis=1))
    den   = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2) + 1e-10
    return num / den


def compute_fundamental(pts1, pts2, use_ransac=True,
                        threshold_px=1.5, iterations=1000):
    # Pasamos de float32 a float64
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    N    = len(pts1)
    assert N >= 8, "Se necesitan al menos 8 correspondencias."

    # Si se quiere ver el resultado obtenido sin utilizar ransac
    # Pero nosotros no lo hacemos, pues el resultado es muy volatil
    if not use_ransac:
        F = _fundamental_8point(pts1, pts2)
        return F, np.ones(N, dtype=bool)

    rng        = np.random.default_rng(42)
    best_F     = None
    best_mask  = np.zeros(N, dtype=bool)
    best_count = 0

    for _ in range(iterations):
        # Escogemos 8 puntos aleatorios
        idx = rng.choice(N, 8, replace=False)
        try:
            # Intentamos calcular la matriz F
            F_cand = _fundamental_8point(pts1[idx], pts2[idx])
        except Exception:
            continue
        # Calculamos su residuo epipolar
        # Distancia entre el punto detectado en la imagen y su línea epipolar
        # res un array de residuos, uno para cada correspondencia
        res     = _epipolar_residuals(F_cand, pts1, pts2)

        # Guardamos en inliers aquellos errores que aceptamos
        # Si su cantidad de puntos cuyos errores aceptamos es mayor
        # que el de la mejor F del momento, la cambiamos
        inliers = res < threshold_px
        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_mask  = inliers
            best_F     = F_cand

    # mask son aquellos puntos "buenos" cuyos residuos aceptamos
    # Estos pueden ser más de 8 puntos, y puede no tener alguno de los
    # 8 aleatoriamente escogidos
    if best_mask.sum() >= 8:
        # Volvemos a calcular la matriz fundamental, pero ahora solo con
        # aquellos puntos buenos
        best_F = _fundamental_8point(pts1[best_mask], pts2[best_mask])

    return best_F, best_mask


# ============================================================
# 2. EPIPOLES
# ============================================================

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


# ============================================================
# 3. RECTIFICACIÓN
# ============================================================

def _H_align_epipole(e, altura, anchura):
    """
    Homografía de Hartley que lleva el epipolo al punto en el infinito
    [1, 0, 0], haciendo que todas las líneas epipolares sean horizontales.
 
    Los tres pasos son:

    1. T — traslada el centro de la imagen al origen.
 
    2. R — rota para que el epipolo, ahora expresado relativo al centro,
       quede alineado con el eje X positivo: e_rotado = (f, 0, 1).
 
    3. G — transformación proyectiva que manda (f,0,1) al infinito (f,0,0):
           G = [[1, 0, 0],
                [0, 1, 0],
                [-1/f, 0, 1]]
 
    H = G · R · T
    """
    # Normalizamos con cuidado de no romper la geometría si
    # el punto ya se encuentra en el infinito
    if abs(e[2]) > 1e-10:
        e = e / e[2]

    # Paso 1: trasladar el centro de imagen al origen
    cx, cy = anchura / 2.0, altura / 2.0
    T = np.array([[1, 0, -cx],
                  [0, 1, -cy],
                  [0, 0,   1]], dtype=float)
 
    # Obtengo un epipolo con coordenadas relativas al centro
    e_t = T @ e

    # Paso 2: rotar para alinear e_t con el eje X
    r = np.hypot(e_t[0], e_t[1])
    if abs(e_t[1]) < 1e-9 and e_t[0] > 0:
        # El epipolo ya está sobre el eje X, no hace falta rotar
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=float)
    else:
        cos_a, sin_a = e_t[0] / r, e_t[1] / r
        R = np.array([[ cos_a, sin_a, 0],
                      [-sin_a, cos_a, 0],
                      [     0,     0, 1]], dtype=float)
 
    # Tras R, el epipolo está en (f, 0, 1) con f = r (distancia al centro)
    f = r
 
    # Paso 3: G manda (f,0,1) al infinito (f,0,0)
    G = np.array([[1,      0, 0],
                  [0,      1, 0],
                  [-1.0/f, 0, 1]], dtype=float)
 
    return G @ R @ T


def _fit_to_canvas(H, h, w):
    """
    Calcula una similitud S (escala uniforme + traslación) tal que
    S @ H mapea las 4 esquinas de la imagen (h, w) dentro del canvas.

    Esto soluciona el caso en que el epipolo está dentro de la imagen:
    la H de Hartley pura manda píxeles fuera del canvas y la imagen
    warpeada queda completamente negra.
    """
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [0, h, 1],
                        [w, h, 1]], dtype=float).T       # (3, 4)
    warped  = H @ corners
    warped /= warped[2]
    xs, ys  = warped[0], warped[1]

    # Escala uniforme mínima para que todo el contenido quepa
    scale = min(w / (xs.max() - xs.min() + 1e-10),
                h / (ys.max() - ys.min() + 1e-10))

    # Traslación para centrar en el canvas
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    tx = w / 2 - scale * cx
    ty = h / 2 - scale * cy

    S = np.array([[scale, 0,     tx],
                  [0,     scale, ty],
                  [0,     0,      1]], dtype=float)
    return S


def compute_rectification(F, img_shape):
    """
    Calcula H1, H2 para rectificar el par estéreo.

    Tras aplicar H1 e H2 a las imágenes izquierda y derecha
    respectivamente, las líneas epipolares quedan horizontales en
    ambas imágenes, y la búsqueda de correspondencias se reduce a
    comparar píxeles de la misma fila.

    Parámetros
    ----------
    F          : (3,3) matriz fundamental.
    img_shape  : (h, w) resolución de las imágenes.
    """
    h, w = img_shape[:2]

    e1 = epipole(F.T)   # epipolo en imagen izquierda
    e2 = epipole(F)     # epipolo en imagen derecha

    # Homografías(matrices 3x3 que deforman la imagen)
    # Convierten las líneas epipolares en horizontales(luego paralelas)
    # Esto lo hace mandando el epipolo al infinito
    H1_raw = _H_align_epipole(e1, h, w)
    H2_raw = _H_align_epipole(e2, h, w)

    # Ajustar cada H para que el resultado llene el canvas
    S1 = _fit_to_canvas(H1_raw, h, w)
    S2 = _fit_to_canvas(H2_raw, h, w)

    H1 = S1 @ H1_raw
    H2 = S2 @ H2_raw

    return H1, H2


# ============================================================
# 4. WARPEO DE IMAGEN (nearest-neighbor vectorizado)
# ============================================================

def warp_image(img, H):
    """ 
    Aplicamos a las imágenes la homografía que lleva el epipolo al infinito
    [1,0,0]. De esta manera, las líneas epipolares son rectas horizontales
    y la correspondencia de un punto se puede encontrar en una recta de 1D
    en la otra imagen, en vez de una recta 2D
    """
    # Obtenemos las dimensiones de la imagen
    h, w   = img.shape[:2]

    # Invertimos la homografía.
    # Buscamos mapear todo el espacio de salida(imagen rectificada)
    # Luego en vez de multiplicar los puntos de la imagen por la homografía
    # (lo que podría resultar en manchas negras vacias), preguntamos 
    # qué punto de la imagen original debe ir en cada posición
    H_inv  = np.linalg.inv(H)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    ones   = np.ones_like(xs)

    # Convertimos a coordenadas homogeneas y aplanamos para operar en bloque
    pts    = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T

    # Aplicamos la homografía inversa y normalizamos
    src    = H_inv @ pts
    src   /= src[2]

    # Redondeamos a píxeles, no puntos
    x_src  = np.round(src[0]).astype(np.int32).reshape(h, w)
    y_src  = np.round(src[1]).astype(np.int32).reshape(h, w)
    valid  = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)

    # Ponemos de fondo una imagen en negro y encima la imagen resultado
    out           = np.zeros_like(img)
    out[valid]    = img[y_src[valid], x_src[valid]]
    return out


# ============================================================
# 5. DISPARIDAD (block matching SSD vectorizado)
# ============================================================

def compute_disparity(img1, img2, max_disp=192, block_size=21):
    """
    Block matching por SSD con ventana 2-D.
    El padding asimétrico garantiza que cost_full tenga exactamente
    el mismo shape que la imagen de entrada, para cualquier block_size.

    Asume que las imágenes ya han sido rectificadas (homografía aplicada), por lo que
    el epipolo está en el infinito y la búsqueda de correspondencias se limita a
    desplazamientos puramente horizontales en la misma fila.
    """
    # 1. Preparación de estructuras
    # Dimensiones de la imagen
    h, w  = img1.shape[:2]

    # Convertimos a float para evitar desbordamiento de ints
    i1    = img1.astype(np.float32)
    i2    = img2.astype(np.float32)

    # Radio del bloque
    radioB = block_size // 2

    # Matriz donde almacenaremos el menor error
    # Inicializada a infinito
    best_cost = np.full((h, w), np.inf, dtype=np.float32)
    
    # Matriz donde almacenaremos el resultado final
    disp_map  = np.zeros((h, w), dtype=np.float32)

    # 2. Suma optimizada
    def box_sum(arr):
        """
        Suma de manera eficiente los valores dentro de una ventana móvil 
        de block_size x block_size. En lugar de usar bucles 'for' 
        anidados, usa sumas acumuladas (O(1) por píxel).
        """
        # Suma acumulada vertical
        s = np.cumsum(arr, axis=0)

        # Suma vertical del bloque
        s = s[block_size:] - s[:-block_size]

        # Suma acumulada horizontal
        s = np.cumsum(s, axis=1)

        # Suma horizontal del bloque
        s = s[:, block_size:] - s[:, :-block_size]
        return s

    # 3. Busqueda en la línea epipolar horizontal
    for d in range(max_disp):
        # Lienzo vacio del tamaño de la imagen
        shifted        = np.zeros_like(i2)
        # Alineamos los puntos con disparidad d de ambas imágenes
        shifted[:, d:] = i2[:, :w - d] if d > 0 else i2

        # Calculamos el error
        cost  = box_sum((i1 - shifted) ** 2)

        # La matriz cost es más pequeña que la imagen pues 
        # box_sum pierde los bordes. Calculamos cuanto se redujo
        dh = h - cost.shape[0]
        dw = w - cost.shape[1]

        # Ponemos cost al tamaño original (h,v)
        # Rellenamos duplicando los valores del borde
        cost_full = np.pad(cost,
                           ((radioB, dh - radioB), (radioB, dw - radioB)),
                           mode='edge')
        if d in [0, 8, 16, 32, 48, 63, 100, 140, 192]:
            print(d, np.mean(cost_full))

        # Guardamos qué píxeles donde se ha mejorado el coste
        better            = cost_full < best_cost
        # Actualizamos en el registro el nuevo coste mínimo
        best_cost[better] = cost_full[better]
        # Guardamos en esos píxeles la disparidad d que mejoro el coste
        disp_map[better]  = d

    return disp_map


# ============================================================
# 6. PROFUNDIDAD RELATIVA
# ============================================================

def disparity_to_depth(disp, baseline=1.0, focal=1.0, min_disp=0.5):
    """
    Convierte disparidad a profundidad: Z = f·B / d.

    Los píxeles con d < min_disp no tienen información de correspondencia
    y se marcan con 0 en lugar de convertirse a un valor de profundidad
    enorme (1/1e-6 = 1_000_000) que aplastaría el rango de visualización.

    Si se conocen:
    Baseline: Distancia entre los centros ópticos de las cámaras
    Distancia focal: Distancia desde el centro óptico hasta el plano
    de la imagen(medida en píxeles)
    Se podría calcular una profundidad real, pero sin conocerlos,
    la profundidad calculada es relativa

    Parámetros
    ----------
    min_disp : float — disparidad mínima considerada válida (px).
                       El block matcher devuelve 0 en zonas sin textura
                       o donde no encontró correspondencia.
    """
    # Llenamos el mapa de profundidad con ceros
    depth = np.zeros_like(disp, dtype=np.float32)
    # Encontramos aquellos que sí tienen información de correspondencia
    valid = disp >= min_disp
    # Modificamos el valor de esto últimos con su profundidad
    depth[valid] = (focal * baseline) / disp[valid]
    return depth


# ============================================================
# 7. PIPELINE COMPLETO
# ============================================================

def stereo_depth(imgL, imgR, pts1, pts2,
                 use_ransac=True, threshold_px=1.5, ransac_iters=1000,
                 max_disp=64, block_size=9):
    """
    Pipeline completo: correspondencias -> F -> rectificación -> disparidad -> profundidad.

    Returns
    -------
    disp, depth, imgLr, imgRr, F, mask
    """
    # Calculamos la matriz fundamental y aquellos puntos "buenos"
    # Los puntos buenos son aquellos cuyo error epipolar aceptamos
    F, mask = compute_fundamental(
        pts1, pts2,
        use_ransac=use_ransac,
        threshold_px=threshold_px,
        iterations=ransac_iters,
    )

    # Pasar img_shape para que la rectificación ajuste al canvas
    H1, H2 = compute_rectification(F, imgL.shape)

    imgLr = warp_image(imgL, H1)
    imgRr = warp_image(imgR, H2)

    disp  = compute_disparity(imgLr, imgRr, max_disp=max_disp, block_size=block_size)
    depth = disparity_to_depth(disp)

    print("Disparidades")
    print(disp.min())
    print(disp.max())
    print(np.percentile(disp, 95))
    print(np.percentile(disp, 99))

    pts1h = np.hstack([pts1, np.ones((len(pts1),1))])
    pts2h = np.hstack([pts2, np.ones((len(pts2),1))])

    p1 = (H1 @ pts1h.T).T
    p2 = (H2 @ pts2h.T).T

    p1 /= p1[:,2:3]
    p2 /= p2[:,2:3]

    print("error vertical medio:", np.mean(np.abs(p1[:,1]-p2[:,1])))
    print("error horizontal medio:", np.mean(np.abs(p1[:,0]-p2[:,0])))

    print("epipolo izquierdo:", epipole(F))
    print("epipolo izquierdo:", epipole(F.T))

    err_y = np.abs(p1[:,1] - p2[:,1])

    print("Error vertical medio:", err_y.mean())
    print("Error vertical máximo:", err_y.max())

    valid = disp > 0

    print("min:", disp[valid].min())
    print("max:", disp[valid].max())
    print("media:", disp[valid].mean())

    return disp, depth, imgLr, imgRr, F, mask

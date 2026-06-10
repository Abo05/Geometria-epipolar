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
                        threshold_px=0.5, iterations=5000):
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


def _fit_shared(H1_raw, H2_raw, h, w):
    """
    Calcula S1 y S2 con escala y traslación-Y COMPARTIDAS entre ambas
    homografías, y traslación-X independiente para centrar cada imagen.
 
    Por qué escala y ty deben ser iguales:
    H1 y H2 alinean las líneas epipolares con el eje X, pero la
    transformación proyectiva G introduce distorsión diferente en cada
    imagen según la distancia del epipolo al centro. Si S1 y S2 se
    calculan independientemente, cada una rescala su imagen a su manera:
    una fila y=300 en imgLr no corresponde a y=300 en imgRr, y el block
    matcher busca en la fila equivocada → error vertical grande → mapa
    de disparidad incorrecto.
 
    La solución: escala = mínimo de ambas (para que las dos quepan),
    ty = centrado sobre el rango Y combinado de ambas imágenes.
    Solo tx puede diferir: la diferencia horizontal entre los campos
    de visión de cada cámara es precisamente la disparidad que queremos medir.
    """
    def bbox(H):
        corners = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]], dtype=float).T
        warped  = H @ corners
        warped /= warped[2]
        return warped[0], warped[1]
 
    xs1, ys1 = bbox(H1_raw)
    xs2, ys2 = bbox(H2_raw)
 
    # Escala compartida: la más pequeña de las cuatro restricciones
    scale = min(w / (xs1.max()-xs1.min()+1e-10),
                h / (ys1.max()-ys1.min()+1e-10),
                w / (xs2.max()-xs2.min()+1e-10),
                h / (ys2.max()-ys2.min()+1e-10))
 
    # ty compartido: centrar el rango Y combinado en el canvas
    y_mid = (min(ys1.min(), ys2.min()) + max(ys1.max(), ys2.max())) / 2
    ty    = h / 2 - scale * y_mid
 
    # tx independiente: centrar cada imagen en X por separado
    tx1 = w / 2 - scale * (xs1.min() + xs1.max()) / 2
    tx2 = w / 2 - scale * (xs2.min() + xs2.max()) / 2
 
    S1 = np.array([[scale, 0, tx1], [0, scale, ty], [0, 0, 1]], dtype=float)
    S2 = np.array([[scale, 0, tx2], [0, scale, ty], [0, 0, 1]], dtype=float)
    return S1, S2


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

    # S1 y S2 compartidas: misma escala y mismo ty para preservar
    # la alineación vertical que H1_raw y H2_raw ya garantizan.
    S1, S2 = _fit_shared(H1_raw, H2_raw, h, w)

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

def compute_disparity(img1, img2, max_disp=64, block_size=9,
                      uniqueness=0.15, median_ksize=5):
    """
    Block matching por SSD con ventana 2-D y post-procesado.
 
    Asume imágenes rectificadas: la correspondencia de un píxel (x,y)
    de img1 solo puede estar en la fila y de img2, desplazada d píxeles
    a la izquierda (d ∈ [0, max_disp)).
 
    La dirección del desplazamiento:
    La cámara derecha está a la derecha de la izquierda, así que el mismo
    objeto aparece más a la IZQUIERDA en img2. Para alinear img2 con img1
    se desplaza img2 d píxeles a la DERECHA: shifted[:, d:] = img2[:, :w-d].
    Esto es equivalente a comparar img1[x] con img2[x-d] para cada x. ✓
 
    Fuentes de ruido y sus correcciones:
 
    PROBLEMA 1 — Mínimo ambiguo en zonas sin textura:
    En zonas uniformes, el SSD es casi idéntico para todas las disparidades
    y d=0 gana por ser el primero. Produce grandes manchas de disparidad 0.
    Corrección: test de unicidad — solo se acepta una disparidad si su coste
    es significativamente menor que el segundo mejor (ratio < 1-uniqueness).
 
    PROBLEMA 2 — Outliers aislados en bordes y zonas de baja textura:
    El mínimo del SSD puede saltar entre píxeles vecinos produciendo ruido
    sal-y-pimienta. No se puede eliminar durante el matching sin perder bordes.
    Corrección: filtro de mediana sobre el mapa final. La mediana preserva
    los bordes mejor que un filtro de media o gaussiano porque usa el valor
    de un vecino real en lugar de promediar.
 
    Parameters
    ----------
    uniqueness   : float — margen mínimo entre el mejor y segundo mejor coste,
                           expresado como fracción del mejor coste.
                           0.15 significa que el segundo mejor debe ser al menos
                           15% peor que el mejor para aceptar la disparidad.
                           Mayor valor → más estricto, más píxeles inválidos.
    median_ksize : int   — tamaño del kernel del filtro de mediana (impar).
                           0 desactiva el filtro.
    """
    h, w  = img1.shape[:2]
    i1    = img1.astype(np.float32)
    i2    = img2.astype(np.float32)
    pad   = block_size // 2
 
    # Guardamos los dos mejores costes para el test de unicidad
    best_cost        = np.full((h, w), np.inf, dtype=np.float32)
    second_best_cost = np.full((h, w), np.inf, dtype=np.float32)
    disp_map         = np.zeros((h, w), dtype=np.float32)
 
    def box_sum(arr):
        """
        Suma en ventana móvil block_size×block_size usando sumas acumuladas.
        Coste O(1) por píxel independientemente del tamaño del bloque.
        """
        s = np.cumsum(arr, axis=0)
        s = s[block_size:] - s[:-block_size]        # suma vertical del bloque
        s = np.cumsum(s, axis=1)
        s = s[:, block_size:] - s[:, :-block_size]  # suma horizontal del bloque
        return s
 
    for d in range(max_disp):
        shifted        = np.zeros_like(i2)
        shifted[:, d:] = i2[:, :w - d] if d > 0 else i2
 
        cost = box_sum((i1 - shifted) ** 2)
 
        # Padding asimétrico: recupera exactamente (h, w) para cualquier
        # block_size y resolución de imagen
        dh = h - cost.shape[0]
        dw = w - cost.shape[1]
        cost_full = np.pad(cost,
                           ((pad, dh - pad), (pad, dw - pad)),
                           mode='edge')
 
        # Actualizar los dos mejores costes
        is_best   = cost_full < best_cost
        is_second = (~is_best) & (cost_full < second_best_cost)
 
        second_best_cost[is_best]   = best_cost[is_best]   # el anterior mejor pasa a segundo
        second_best_cost[is_second] = cost_full[is_second]
 
        best_cost[is_best] = cost_full[is_best]
        disp_map[is_best]  = d
 
    # Test de unicidad: descarta píxeles donde el mejor y segundo mejor
    # coste son demasiado parecidos (mínimo poco pronunciado → match ambiguo)
    if uniqueness > 0:
        ambiguous          = best_cost * (1.0 + uniqueness) > second_best_cost
        disp_map[ambiguous] = 0
 
    # Filtro de mediana: elimina outliers sal-y-pimienta preservando bordes
    if median_ksize > 1:
        disp_map = cv2.medianBlur(disp_map.astype(np.float32), median_ksize)
 
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

# ===========================
# BORRAR
# ===========================
def draw_epipolar_lines_opencv(img1, img2, F, pts1, pts2, num_lines=10):
    """
    Dibuja líneas epipolares usando cv2.computeCorrespondEpilines.
    - En img1: líneas epipolares de los puntos de img2.
    - En img2: líneas epipolares de los puntos de img1.
    """
    img1_line = img1.copy()
    img2_line = img2.copy()
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Seleccionar algunos puntos aleatorios
    n = min(len(pts1), num_lines)
    idx = np.random.choice(len(pts1), n, replace=False)
    pts1_sample = pts1[idx].reshape(-1, 1, 2).astype(np.float32)
    pts2_sample = pts2[idx].reshape(-1, 1, 2).astype(np.float32)
    
    # Líneas en img1 a partir de puntos en img2
    lines1 = cv2.computeCorrespondEpilines(pts2_sample, 2, F)
    lines1 = lines1.reshape(-1, 3)
    for l, pt1 in zip(lines1, pts1_sample.reshape(-1, 2)):
        a, b, c = l
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
        x1, y1 = w1, int((-c - a*w1) / b) if abs(b) > 1e-6 else h1
        if 0 <= y0 < h1 and 0 <= y1 < h1:
            cv2.line(img1_line, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(img1_line, tuple(pt1.astype(int)), 4, (0, 0, 255), -1)
    
    # Líneas en img2 a partir de puntos en img1
    lines2 = cv2.computeCorrespondEpilines(pts1_sample, 1, F)
    lines2 = lines2.reshape(-1, 3)
    for l, pt2 in zip(lines2, pts2_sample.reshape(-1, 2)):
        a, b, c = l
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
        x1, y1 = w2, int((-c - a*w2) / b) if abs(b) > 1e-6 else h2
        if 0 <= y0 < h2 and 0 <= y1 < h2:
            cv2.line(img2_line, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(img2_line, tuple(pt2.astype(int)), 4, (0, 0, 255), -1)
    
    return img1_line, img2_line

def draw_horizontal_lines(imgLr, imgRr, pts1_rect, pts2_rect, num_lines=10):
    """
    Dibuja líneas horizontales en imágenes rectificadas para verificar
    que las correspondencias están en la misma fila.
    """
    h, w = imgLr.shape[:2]
    n = min(len(pts1_rect), num_lines)
    idx = np.random.choice(len(pts1_rect), n, replace=False)
    
    imgL_out = imgLr.copy()
    imgR_out = imgRr.copy()
    
    for i in idx:
        x1, y1 = pts1_rect[i]
        x2, y2 = pts2_rect[i]
        y = int(round(y1))
        if 0 <= y < h:
            cv2.line(imgL_out, (0, y), (w-1, y), (255, 0, 0), 1)
            cv2.line(imgR_out, (0, y), (w-1, y), (255, 0, 0), 1)
        cv2.circle(imgL_out, (int(x1), int(y1)), 4, (0, 0, 255), -1)
        cv2.circle(imgR_out, (int(x2), int(y2)), 4, (0, 0, 255), -1)
    return imgL_out, imgR_out

# ===========================

# ============================================================
# 7. PIPELINE COMPLETO
# ============================================================

def stereo_depth(imgL, imgR, pts1, pts2,
                 use_ransac=True, threshold_px=1.5, ransac_iters=1000,
                 max_disp=256, block_size=9):
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

        # Dibujar líneas epipolares en imágenes originales
    img1_epi, img2_epi = draw_epipolar_lines_opencv(imgL, imgR, F, pts1, pts2, num_lines=15)

    # Pasar img_shape para que la rectificación ajuste al canvas
    H1, H2 = compute_rectification(F, imgL.shape)

    imgLr = warp_image(imgL, H1)
    imgRr = warp_image(imgR, H2)

        # Aplicar las homografías a los puntos inliers para verificar alineación horizontal
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))]).T
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))]).T
    pts1_rect = (H1 @ pts1_h).T
    pts2_rect = (H2 @ pts2_h).T
    pts1_rect /= pts1_rect[:, 2:3]
    pts2_rect /= pts2_rect[:, 2:3]

    # Dibujar líneas horizontales en imágenes rectificadas
    imgL_horiz, imgR_horiz = draw_horizontal_lines(imgLr, imgRr, pts1_rect[:, :2], 
                                                   pts2_rect[:, :2], num_lines=15)

    # Mostrar resultados
    cv2.imwrite('output/epipolaresIzquierda.png', img1_epi)
    cv2.imwrite('output/epipolaresDerecha.png', img2_epi)
    cv2.imwrite('output/rectificadaIzquierda.png', imgL_horiz)
    cv2.imwrite('output/rectificadaDerecha.png', imgR_horiz)

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

    err_y = np.abs(p1[:,1] - p2[:,1])

    print("Error vertical máximo:", err_y.max())

    valid = disp > 0

    print("min:", disp[valid].min())
    print("max:", disp[valid].max())
    print("media:", disp[valid].mean())

    return disp, depth, imgLr, imgRr, F, mask

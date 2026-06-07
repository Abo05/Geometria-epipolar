"""
Cálculo de la Matriz Fundamental F — Algoritmo de los 8 puntos normalizado
Uso:
    F, mask = compute_fundamental_matrix(pts_left, pts_right)

donde pts_left y pts_right son arrays (N, 2) de correspondencias en píxeles.
"""

import numpy as np
import cv2  # Solo para detectar correspondencias; F se calcula desde cero


# ---------------------------------------------------------------------------
# 1. NORMALIZACIÓN
# ---------------------------------------------------------------------------

def compute_normalization_matrix(pts):
    """
    Calcula la transformación de normalización T tal que x' = T·x.

    Criterio:
      - El centroide de los puntos normalizados debe estar en el origen.
      - La distancia RMS de los puntos al origen debe ser √2.
    Returns:
        T: array (3, 3) — matriz de normalización homogénea.
    """
    centroid = pts.mean(axis=0)                      # (cx, cy)
    pts_centered = pts - centroid

    # Distancia RMS al origen -> factor de escala para que RMS = sqrt(2)
    rms_dist = np.sqrt((pts_centered ** 2).sum(axis=1).mean())
    scale = np.sqrt(2) / rms_dist

    # T traslada al centroide y luego escala
    # T = [[s, 0, -s·cx],
    #      [0, s, -s·cy],
    #      [0, 0,   1  ]]
    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,         0,                    1],
    ])
    return T


def normalize_points(pts, T):
    """
    Aplica la transformación homogénea T a un array de puntos 2D.
    Returns:
        pts_norm: array (N, 2) — puntos normalizados.
    """
    N = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N, 1))])        # (N, 3) coordenadas homogéneas
    pts_norm_h = (T @ pts_h.T).T                     # (N, 3)
    return pts_norm_h[:, :2]                          # volver a (N, 2)


# ---------------------------------------------------------------------------
# 2. CONSTRUCCIÓN DE LA MATRIZ A
# ---------------------------------------------------------------------------

def build_design_matrix(pts1_norm, pts2_norm):
    """
    Construye la matriz de diseño A tal que A·f = 0,
    donde f son los 9 elementos de F en orden fila-mayor.
    Returns:
        A: array (N, 9).
    """
    x  = pts1_norm[:, 0]
    y  = pts1_norm[:, 1]
    xp = pts2_norm[:, 0]
    yp = pts2_norm[:, 1]
    ones = np.ones(len(x))

    # Columnas: x'x  x'y  x'  y'x  y'y  y'  x  y  1
    A = np.column_stack([
        xp * x,  xp * y,  xp,
        yp * x,  yp * y,  yp,
        x,       y,       ones,
    ])
    return A


# ---------------------------------------------------------------------------
# 3. SOLUCIÓN LINEAL VIA SVD
# ---------------------------------------------------------------------------

def solve_f_from_svd(A):
    """
    Resuelve A·f = 0 mediante SVD.

    La solución f es el vector singular derecho correspondiente al menor
    valor singular de A, es decir, la última columna de V en A = UDVᵀ.

    Returns:
        F_hat: array (3, 3) — estimación inicial de F (aún no es rango 2).
    """
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1]                                        # último vector singular derecho
    F_hat = f.reshape(3, 3)
    return F_hat


# ---------------------------------------------------------------------------
# 4. IMPOSICIÓN DEL RANGO 2
# ---------------------------------------------------------------------------

def enforce_rank2(F_hat):
    """
    Reemplaza F por la matriz singular más cercana en norma de Frobenius.

    Returns:
        F_rank2: array (3, 3) — estimación con rango 2 impuesto.
    """
    U, S, Vt = np.linalg.svd(F_hat)
    S[2] = 0                                          # zeroing del menor valor singular
    F_rank2 = U @ np.diag(S) @ Vt
    return F_rank2


# ---------------------------------------------------------------------------
# 5. DESNORMALIZACIÓN
# ---------------------------------------------------------------------------

def denormalize(F_hat_rank2, T, Tp):
    """
    Lleva F del espacio normalizado al espacio original de píxeles.

    IMPORTANTE: H&Z recomiendan imponer rango 2 *antes* de desnormalizar

    Returns:
        F: array (3, 3) — matriz fundamental en coordenadas originales.
    """
    F = Tp.T @ F_hat_rank2 @ T
    return F


# ---------------------------------------------------------------------------
# 6. ALGORITMO COMPLETO
# ---------------------------------------------------------------------------

def compute_fundamental_matrix(pts1, pts2):
    """
    Calcula la matriz fundamental F usando el algoritmo de los 8 puntos
    normalizado de Hartley & Zisserman (Algoritmo 11.1, p. 282).

    Pasos:
        (i)  Normalización
        (ii)(a) Solución lineal
        (ii)(b) Imposición rango 2
        (iii) Desnormalización

    Returns:
        F: array (3, 3) — matriz fundamental normalizada.
           Satisface: pts2[i] (homog.) @ F @ pts1[i] (homog.) ≈ 0
    """
    assert len(pts1) >= 8, "Se necesitan al menos 8 correspondencias."
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)

    # Paso (i): Normalización
    T  = compute_normalization_matrix(pts1)
    Tp = compute_normalization_matrix(pts2)
    pts1_norm = normalize_points(pts1, T)
    pts2_norm = normalize_points(pts2, Tp)

    # Paso (ii)(a): Construir A y resolver Af = 0
    A = build_design_matrix(pts1_norm, pts2_norm)
    F_hat = solve_f_from_svd(A)

    # Paso (ii)(b): Imponer rango 2 ANTES de desnormalizar
    F_hat_rank2 = enforce_rank2(F_hat)

    # Paso (iii): Desnormalizar
    F = denormalize(F_hat_rank2, T, Tp)

    # Normalizar por el último elemento para consistencia de escala
    F = F / F[2, 2]

    return F


# ---------------------------------------------------------------------------
# 7. VERIFICACIÓN Y MÉTRICAS DE CALIDAD
# ---------------------------------------------------------------------------

def epipolar_residual(F, pts1, pts2):
    """
    Calcula el error epipolar medio: d(x', l') donde l' = F·x.

    Para cada correspondencia calcula la distancia del punto x' a su línea
    epipolar l' = F·x (en la imagen derecha). Un buen F da residuos < 1px.

    Returns:
        mean_residual: float — error medio en píxeles.
        residuals:     array (N,) — error por correspondencia.
    """
    N = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((N, 1))])      # (N, 3)
    pts2_h = np.hstack([pts2, np.ones((N, 1))])      # (N, 3)

    # Líneas epipolares en imagen derecha: l' = F·x
    lines = (F @ pts1_h.T).T                         # (N, 3) — coeficientes (a, b, c)

    # Distancia punto-línea: |a·x' + b·y' + c| / sqrt(a² + b²)
    num = np.abs((pts2_h * lines).sum(axis=1))
    den = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
    residuals = num / den

    return residuals.mean(), residuals


def check_epipole_constraint(F):
    """
    Verifica que F tiene rango 2 comprobando det(F) ≈ 0.

    Returns:
        det_val: float — debería ser cercano a 0.
    """
    return np.linalg.det(F)


# ---------------------------------------------------------------------------
# 8. CALCULA FINALMENTE F CON RANSAC
# ---------------------------------------------------------------------------

def find_correspondences(img_left, img_right, max_features=500):
    """
    Detecta correspondencias SIFT entre dos imágenes.
    (Esta parte puede quedarse en OpenCV — el matching no es geometría epipolar)

    Returns:
        pts1, pts2: arrays (N, 2) de correspondencias filtradas por ratio test.
    """
    sift = cv2.SIFT_create(max_features)
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test de Lowe (retiene solo matches inequívocos)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def compute_F_with_ransac(pts1, pts2, threshold_px=1.0, iterations=1000):
    """
    Calcula F con RANSAC para robustez ante outliers (H&Z sección 11.6,
    Algoritmo 11.4, p. 291).

    En cada iteración selecciona 8 correspondencias aleatorias, calcula F
    con el algoritmo de los 8 puntos normalizado, y cuenta los inliers
    (puntos cuyo error epipolar < threshold_px).

    Returns:
        F_best:  array (3, 3) — mejor F encontrada.
        mask:    array (N,) bool — True para inliers.
    """
    N = len(pts1)
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    F_best = None
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        # Muestra mínima: 8 correspondencias (H&Z p. 279)
        idx = rng.choice(N, 8, replace=False)
        try:
            F_candidate = compute_fundamental_matrix(pts1[idx], pts2[idx])
        except Exception:
            continue

        # Contar inliers por error epipolar
        _, residuals = epipolar_residual(F_candidate, pts1, pts2)
        inliers = residuals < threshold_px

        if inliers.sum() > best_count:
            best_count = inliers.sum()
            best_inliers = inliers
            F_best = F_candidate

    # Refinamiento final con todos los inliers
    if best_inliers.sum() >= 8:
        F_best = compute_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])

    return F_best, best_inliers


# ---------------------------------------------------------------------------
# 9. EJEMPLO DE USO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    img_left  = cv2.imread("im0.png", cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread("im1.png", cv2.IMREAD_GRAYSCALE)

    if img_left is None or img_right is None:
        print("Coloca im0.png e im1.png en el directorio actual.")
        sys.exit(1)

    print("Detectando correspondencias SIFT...")
    pts1, pts2 = find_correspondences(img_left, img_right)
    print(f"  {len(pts1)} correspondencias candidatas encontradas.")

    print("Calculando F con RANSAC + algoritmo 8 puntos normalizado...")
    F, mask = compute_F_with_ransac(pts1, pts2, threshold_px=1.0)
    print(f"  Inliers: {mask.sum()} / {len(mask)}")

    mean_err, _ = epipolar_residual(F, pts1[mask], pts2[mask])
    print(f"  Error epipolar medio (inliers): {mean_err:.4f} px")
    print(f"  det(F) = {check_epipole_constraint(F):.2e}  (debe ser ≈ 0)")
    print(f"\nMatriz fundamental F:\n{F}")

    # Comparación con OpenCV (referencia)
    F_cv, mask_cv = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    mean_err_cv, _ = epipolar_residual(F_cv, pts1, pts2)
    print(f"\n[Referencia OpenCV] Error epipolar medio: {mean_err_cv:.4f} px")

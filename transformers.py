import cv2
import numpy as np
import math
import logging
from typing import List
from typing import Tuple

logger = logging.getLogger(__name__)

def ensure_proper_size(img: np.ndarray, max_dim_size: int) -> np.ndarray:
    """
    Asegura que la imagen tenga un tamaño adecuado, redimensionando si excede max_dim_size.
    Proveniente de la lógica original de ImagePreprocessor.
    """
    try:
        h, w = img.shape[:2]
        if max_dim_size <= 0:
            logger.warning(f"max_dim_size ({max_dim_size}) no es válido en ensure_proper_size. No se redimensiona.")
            return img
        if max(h, w) > max_dim_size:
            scale = max_dim_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(f"Redimensionando imagen de ({w},{h}) a ({new_w},{new_h}) porque excedía el tamaño máximo de {max_dim_size}")
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        logger.error(f"Error en ensure_proper_size: {e}", exc_info=True)
        return img
    
def high_precision_skew_correction(gray_img: np.ndarray, skew_threshold: float,
                                   canny_thresh1: int, canny_thresh2: int, canny_aperture: int,
                                   hough_min_line_len_cap: int, hough_max_gap: int, hough_thresh: int,
                                   hough_angle_range: List[float]
                                   ) -> Tuple[np.ndarray, float]:
    detected_angle = 0.0
    try:
        logger.debug(f"Iniciando high_precision_skew_correction con umbral: {skew_threshold}, "
                    f"Canny:({canny_thresh1},{canny_thresh2},{canny_aperture}), "
                    f"Hough:(cap={hough_min_line_len_cap},gap={hough_max_gap},thresh={hough_thresh},angle_range={hough_angle_range})")

        edges = cv2.Canny(gray_img, canny_thresh1, canny_thresh2, apertureSize=canny_aperture) # Usar parámetros
        min_line_length_val = min(gray_img.shape[1] // 2, hough_min_line_len_cap) # Usar parámetro

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh, # Usar parámetro
                                minLineLength=min_line_length_val,
                                maxLineGap=hough_max_gap)
        
        if lines is None or len(lines) < 5: # Este umbral de 5 líneas podría ser configurable
            logger.warning("No se detectaron suficientes líneas para la corrección de inclinación precisa.")
            return gray_img, detected_angle

        angles = []
        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Filtrar ángulos usando el parámetro hough_angle_range
            if hough_angle_range[0] < angle < hough_angle_range[1]: # Usar parámetro
                angles.append(angle)
                
        if not angles:
            logger.warning("No se encontraron ángulos de línea adecuados para la corrección de inclinación precisa.")
            return gray_img, detected_angle # Devuelve la imagen original y 0.0

        median_angle = np.median(angles)
        detected_angle = median_angle # Guardar el ángulo detectado
        #logger.info(f"Ángulo de inclinación preciso detectado (mediana): {median_angle:.2f}°")


        if abs(median_angle) > skew_threshold:
          #  logger.info(f"Aplicando corrección de inclinación precisa de {median_angle:.2f}° (umbral: {skew_threshold}°).")
            h, w = gray_img.shape
            center = (w // 2, h // 2)
            # Se rota por 'median_angle' para enderezar la imagen
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0) 
            rotated_img = cv2.warpAffine(gray_img, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
            return rotated_img, median_angle # Devuelve imagen rotada y el ángulo por el cual se rotó
        else:
            logger.info(f"Inclinación precisa de {median_angle:.2f}° por debajo del umbral de {skew_threshold}°. No se aplica corrección.")
            return gray_img, median_angle # Devuelve imagen original y el ángulo detectado (que no superó el umbral)

    except Exception as e:
        logger.error(f"Error en high_precision_skew_correction: {e}", exc_info=True)
        return gray_img, detected_angle # En caso de error, devuelve la imagen original y 0.0

def rotate_image_by_angle(image: np.ndarray, angle_degrees: float, border_mode=cv2.BORDER_REPLICATE) -> np.ndarray:
    """
    Rota una imagen por un ángulo específico.
    Esta es la función canónica que será utilizada por SkewAlignmentOptimizer y otros.
    """
    if abs(angle_degrees) < 0.05: 
        logger.debug(f"Ángulo de rotación {angle_degrees}° demasiado pequeño, no se rota.")
        return image

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        logger.warning("Intentando rotar una imagen vacía.")
        return image

    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    try:
        rotated_img = cv2.warpAffine(image, rotation_matrix, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=border_mode)
        logger.info(f"Imagen rotada {angle_degrees:.2f} grados.")
        return rotated_img
    except Exception as e:
        logger.error(f"Error en rotate_image_by_angle: {e}", exc_info=True)
        return image

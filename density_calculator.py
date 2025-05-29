# PerfectOCR/core/spatial_analysis/density_calculator.py
import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

def calculate_density_map(binary_image_normalized_text_as_1: np.ndarray, 
                          window_size: int) -> Optional[np.ndarray]:
    """
    Calcula el mapa de densidad local D(x,y) usando un filtro de caja (promedio).
    La imagen de entrada debe estar normalizada: texto=1.0, fondo=0.0.
    El tipo de dato de entrada debe ser float para cv2.boxFilter con normalize=True.

    Args:
        binary_image_normalized_text_as_1: Imagen binarizada y normalizada (texto=1, fondo=0)
        window_size: Tamaño de la ventana del filtro (debe ser impar y positivo)

    Returns:
        El mapa de densidad (float32, valores entre 0 y 1) o None si hay error
    """
    if binary_image_normalized_text_as_1 is None or binary_image_normalized_text_as_1.size == 0:
        logger.warning("Mapa de densidad no calculado: imagen normalizada de entrada vacía o None.")
        return None
    
    if not isinstance(binary_image_normalized_text_as_1, np.ndarray):
        logger.error("La entrada para calculate_density_map no es un array de numpy.")
        return None

    if window_size <= 0 or window_size % 2 == 0: 
        logger.error(f"density_map_window_size ({window_size}) no es válido (debe ser impar y >0).")
        return None

    try:
        if binary_image_normalized_text_as_1.dtype != np.float32:
            image_to_filter = binary_image_normalized_text_as_1.astype(np.float32)
        else:
            image_to_filter = binary_image_normalized_text_as_1

        density_map = cv2.boxFilter(
            src=image_to_filter,
            ddepth=-1, 
            ksize=(window_size, window_size),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT
        )
        logger.info(f"Mapa de densidad calculado con ventana de {window_size}x{window_size}.")
        return density_map
    
    except cv2.error as e_cv:
        logger.error(f"Error de OpenCV calculando el mapa de densidad: {e_cv}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error general calculando el mapa de densidad: {e}", exc_info=True)
        return None

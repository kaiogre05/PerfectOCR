import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detail_preserving_denoise(img: np.ndarray, strength: int) -> np.ndarray:
    """
    Reducción de ruido que preserva detalles finos.
    Proveniente de la lógica original de ImagePreprocessor.
    """
    try:
        logger.debug(f"Aplicando detail_preserving_denoise con fuerza: {strength}")
        return cv2.fastNlMeansDenoising(img, None, h=strength,
                                       templateWindowSize=7,
                                       searchWindowSize=21)
    except Exception as e:
        logger.error(f"Error en detail_preserving_denoise: {e}", exc_info=True)
        return img

def high_res_binarization(img: np.ndarray, block_size: int, c_value: int) -> np.ndarray:
    """
    Binarización adaptativa optimizada para alta resolución.
    Proveniente de la lógica original de ImagePreprocessor.
    """
    try:
        current_block_size = block_size
        if current_block_size <= 1:
            current_block_size = 3
            logger.warning(f"block_size en high_res_binarization era <=1, ajustado a {current_block_size}")
        elif current_block_size % 2 == 0:
            current_block_size += 1
            logger.warning(f"block_size en high_res_binarization era par, ajustado a {current_block_size}")
        
        logger.debug(f"Aplicando high_res_binarization con block_size: {current_block_size}, C: {c_value}")
        return cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   current_block_size,
                                   c_value)
    except Exception as e:
        logger.error(f"Error en high_res_binarization: {e}", exc_info=True)
        return img

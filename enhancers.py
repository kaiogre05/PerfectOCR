import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def conservative_contrast_enhancement(img: np.ndarray, clip_limit: float, grid_size: tuple) -> np.ndarray:
    """
    Mejora de contraste optimizada con CLAHE.
    Proveniente de la lógica original de ImagePreprocessor.
    """
    try:
        grid_size_tuple = tuple(map(int, grid_size))
        logger.debug(f"Aplicando conservative_contrast_enhancement con clip_limit: {clip_limit}, grid_size: {grid_size_tuple}")
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size_tuple)
        return clahe.apply(img)
    except Exception as e:
        logger.error(f"Error en conservative_contrast_enhancement: {e}", exc_info=True)
        return img
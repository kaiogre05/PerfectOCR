# PerfectOCR/core/preprocessing/image_corrector.py
import numpy as np
import logging
from core.preprocessing import toolbox
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ImageCorrector:
    def apply_grayscale_corrections(self, gray_image: np.ndarray, correction_plan: Dict[str, Any]) -> np.ndarray:
        """
        Aplica correcciones en escala de grises según el plan proporcionado.
        """
        if not isinstance(correction_plan, dict):
            logger.warning(f"Plan de corrección no es un diccionario: {type(correction_plan)}")
            return gray_image

        corrected_image = gray_image.copy()

        # 1. Corrección de Inclinación
        if 'deskew' in correction_plan and 'angle' in correction_plan['deskew']:
            angle = correction_plan['deskew']['angle']
            if angle != 0.0:
                corrected_image = toolbox.apply_deskew(corrected_image, angle)
        
        # 2. Eliminación de Ruido
        if 'denoise' in correction_plan and 'strength' in correction_plan['denoise']:
            strength = correction_plan['denoise']['strength']
            corrected_image = toolbox.apply_denoise(corrected_image, strength)

        # 3. Mejora de Contraste
        if 'contrast' in correction_plan and 'clahe_params' in correction_plan['contrast']:
            clahe_params = correction_plan['contrast']['clahe_params']
            corrected_image = toolbox.apply_clahe_contrast(corrected_image, **clahe_params)

        return corrected_image
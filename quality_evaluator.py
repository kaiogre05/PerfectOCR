# PerfectOCR/core/input_validation/quality_evaluator.py
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ImageQualityEvaluator:
    def __init__(self, config: Dict):
        # La configuración que recibe es la sección 'quality_assessment_rules'
        self.rules = config

    def _detect_skew_angle(self, gray_image: np.ndarray) -> float:
        deskew_rules = self.rules.get('deskew', {})
        try:
            # Obtener el ángulo máximo permitido desde las reglas
            max_allowed_angle = deskew_rules.get('max_allowed_angle', 1.0)  # Por defecto 1 grado
            
            canny_params = deskew_rules.get('canny_thresholds', [50, 150])
            edges = cv2.Canny(gray_image, canny_params[0], canny_params[1])
            
            min_line_len = min(gray_image.shape[1] // 2, deskew_rules.get('hough_min_line_length_cap_px', 300))
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=deskew_rules.get('hough_threshold', 150),
                                    minLineLength=min_line_len,
                                    maxLineGap=deskew_rules.get('hough_max_line_gap_px', 20))
            if lines is None: return 0.0
            
            angles = [math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
            angle_range = deskew_rules.get('hough_angle_filter_range_degrees', [-20.0, 20.0])
            filtered_angles = [angle for angle in angles if angle_range[0] < angle < angle_range[1]]
            
            detected_angle = np.median(filtered_angles) if filtered_angles else 0.0
            
            # Si el ángulo detectado es mayor que el máximo permitido, aplicar la corrección
            if abs(detected_angle) > max_allowed_angle:
                return detected_angle
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"Error en la detección de inclinación: {e}", exc_info=True)
            return 0.0

    def _detect_noise_level(self, image: np.ndarray) -> float:
        """Detecta el nivel de ruido usando la varianza del Laplaciano."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = np.var(laplacian)
            
            # Normalización basada en umbrales típicos
            min_noise = 50
            max_noise = 1000
            normalized_noise = (noise_level - min_noise) / (max_noise - min_noise)
            return max(0.0, min(1.0, normalized_noise))
            
        except Exception as e:
            logger.error(f"Error en detección de ruido: {e}")
            return 0.0

    def _detect_contrast_level(self, image: np.ndarray) -> float:
        """Detecta el nivel de contraste usando el rango dinámico."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            min_val = np.min(gray)
            max_val = np.max(gray)
            contrast = (max_val - min_val) / 255.0
            return contrast
            
        except Exception as e:
            logger.error(f"Error en detección de contraste: {e}")
            return 1.0

    def evaluate_and_create_correction_plan(self, image: np.ndarray) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        observations = {}
        correction_plans = {}

        for engine in ['tesseract', 'paddleocr']:
            engine_rules = self.rules.get(engine, {})
            plan = {
                'deskew': {'angle': 0.0},
                'denoise': {'strength': 5},
                'contrast': {'clahe_params': {'clip_limit': 1.5, 'grid_size': (8, 8)}}
            }
            engine_obs = []

            # --- Deskew ---
            skew_angle = self._detect_skew_angle(image)
            max_skew = engine_rules.get('deskew', {}).get('max_allowed_angle', 1.0)
            if abs(skew_angle) > max_skew:
                plan['deskew']['angle'] = skew_angle
                engine_obs.append(f"Ángulo de inclinación detectado: {skew_angle:.2f}° (corrigiendo)")

            # --- Ruido ---
            noise_level = self._detect_noise_level(image)
            max_noise = engine_rules.get('denoise', {}).get('max_noise_level', 0.5)
            if noise_level > max_noise:
                plan['denoise']['strength'] = int(min(15, max(5, noise_level * 10)))
                engine_obs.append(f"Ruido detectado: {noise_level:.2f} (corrigiendo)")

            # --- Contraste ---
            contrast_level = self._detect_contrast_level(image)
            min_contrast = engine_rules.get('contrast', {}).get('min_contrast', 1.0)
            if contrast_level < min_contrast:
                plan['contrast']['clahe_params']['clip_limit'] = 2.0
                engine_obs.append(f"Contraste bajo: {contrast_level:.2f} (corrigiendo)")

            correction_plans[engine] = plan
            observations[engine] = engine_obs

        obs_list = [f"[{eng}] {msg}" for eng, msgs in observations.items() for msg in msgs]
        return obs_list, correction_plans
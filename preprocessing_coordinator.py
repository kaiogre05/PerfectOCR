# PerfectOCR/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import os
import logging
import time # Importar el módulo time
from typing import Any, Optional, Dict, List, Tuple
from core.preprocessing import transformers, enhancers, cleaners

logger = logging.getLogger(__name__)

class PreprocessingCoordinator:
    def __init__(self, config: Dict, project_root: str): # config es image_preparation_config
        self.config = config
        self.project_root = project_root
        #logger.info(f"PreprocessingCoordinator inicializado (config image_preparation: {self.config})") #

    def _get_param_from_config(self, config_to_use: Dict, key_path: str, default_value: Any) -> Any:
        keys = key_path.split('.')
        val = config_to_use
        try:
            for key in keys:
                val = val[key]
            return val
        except KeyError:
            return default_value
        except TypeError: 
            return default_value

    def _determine_denoise_strength(self, qa_metrics: Dict, active_config: Dict) -> int:
        denoise_cfg = self._get_param_from_config(active_config, "denoise", {})

        mode = denoise_cfg.get("mode", "fixed") # "fixed" es el default de YAML
        base_strength = int(denoise_cfg.get("fixed_strength", 7)) # 7 es el default de YAML

        if mode == "auto_from_metrics":
            auto_settings = denoise_cfg.get("auto_settings", {})
            thresholds = auto_settings.get("sharpness_thresholds", [50.0, 100.0]) 
            strengths_map = auto_settings.get("strengths_map", [12, base_strength, max(3, base_strength - 2)]) # Defaults de YAML

            sharpness = qa_metrics.get('sharpness_laplacian_variance')
            if sharpness is not None and isinstance(sharpness, (float, int)):
                if sharpness < thresholds[0]:
                    logger.debug(f"Denoise Auto: Nitidez baja ({sharpness:.2f}), usando strength {strengths_map[0]}.")
                    return strengths_map[0]
                elif sharpness < thresholds[1]:
                    logger.debug(f"Denoise Auto: Nitidez media-baja ({sharpness:.2f}), usando strength {strengths_map[1]}.")
                    # Interpretar "base" si es necesario, o YAML debe tener valor numérico
                    return strengths_map[1] if isinstance(strengths_map[1], int) else base_strength
                else:
                    logger.debug(f"Denoise Auto: Nitidez buena ({sharpness:.2f}), usando strength {strengths_map[2]}.")
                    # Interpretar "max(3, base-2)" o YAML debe tener valor numérico
                    if isinstance(strengths_map[2], int): return strengths_map[2]
                    else: return max(3, base_strength - 2) # Mantener lógica si YAML no es explícito
        return base_strength

    def _determine_clahe_tile_grid_size(self, qa_metrics: Dict, active_config: Dict) -> Tuple[int, int]:
        mode = self._get_param_from_config(active_config, "contrast_enhancement.tile_grid_size_mode", "fixed")
        base_grid_list = self._get_param_from_config(active_config, "contrast_enhancement.fixed_tile_grid_size", [8, 8])
        
        if not (isinstance(base_grid_list, list) and len(base_grid_list) == 2 and all(isinstance(x, int) for x in base_grid_list)):
            base_grid_list = [8,8] 
        base_grid_size = tuple(base_grid_list)

        if mode == "auto_from_dimensions":
            dims = qa_metrics.get("dimensions")
            if dims and isinstance(dims, dict):
                h, w = dims.get("height"), dims.get("width")
                if h and w and isinstance(h, (int, float)) and isinstance(w, (int, float)):
                    if h > 2500 or w > 2500: return (12, 12)
                    if h < 1000 or w < 1000: return (6, 6)
        return base_grid_size

    def _determine_binarization_block_size(self, qa_metrics: Dict, active_config: Dict) -> int:
        mode = self._get_param_from_config(active_config, "binarization.block_size_mode", "auto_image_size_heuristic")
        block_size = int(self._get_param_from_config(active_config, "binarization.fixed_block_size", 25))

        if mode == "auto_image_size_heuristic":
            dims = qa_metrics.get("dimensions")
            if dims and isinstance(dims, dict):
                h = dims.get("height")
                if h and isinstance(h, (int, float)):
                    if h < 800: block_size = 11
                    elif h < 1500: block_size = 17
                    elif h < 2500: block_size = 25
                    else: block_size = 35
        
        if block_size <= 0: block_size = 11 
        if block_size % 2 == 0: block_size += 1 
        return block_size

    def apply_preprocessing_pipeline(
        self,
        image_array: np.ndarray, 
        quality_assessment_metrics: Dict[str, Any], 
        image_path_for_log: str 
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        
        stage_start_time = time.perf_counter() # Iniciar temporizador
        active_config = self.config 

        logger.info(f"PreprocessingCoordinator: Iniciando pipeline para {os.path.basename(image_path_for_log)}") 
        logger.debug(f"Métricas de calidad recibidas: {quality_assessment_metrics}")

        # --- 1. Conversión a Escala de Grises (si es necesario) ---
        num_channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
        current_image_gray: Optional[np.ndarray] = None

        if num_channels == 4:
            logger.debug("Convirtiendo imagen BGRA/RGBA a escala de grises.")
            current_image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGRA2GRAY)
        elif num_channels == 3:
            logger.debug("Convirtiendo imagen BGR/RGB a escala de grises.")
            current_image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        elif num_channels == 1:
            logger.debug("La imagen ya está en escala de grises.")
            current_image_gray = image_array.copy() 
        else:
            logger.error(f"Formato de imagen (canales: {num_channels}) no soportado para conversión a gris.")
            elapsed_time = time.perf_counter() - stage_start_time # Calcular tiempo antes de retornar
            return None, elapsed_time 
        if current_image_gray is None: 
            elapsed_time = time.perf_counter() - stage_start_time
            return None, elapsed_time

        # --- 3. Corrección de Inclinación (Deskew) ---
        estimated_angle = quality_assessment_metrics.get('estimated_skew_angle_degrees', 0.0)
        deskew_app_thresh = float(self._get_param_from_config(active_config, 'deskew.application_angle_threshold', 0.5))
        
        gray_image_deskewed = current_image_gray
        actual_angle_corrected = 0.0
        if abs(estimated_angle) > deskew_app_thresh:
            logger.info(f"Aplicando corrección de inclinación de {estimated_angle:.2f}° (umbral: {deskew_app_thresh}°).")
            gray_image_deskewed = transformers.rotate_image_by_angle(current_image_gray, estimated_angle) 
            actual_angle_corrected = estimated_angle
        else:
            logger.info(f"Inclinación estimada de {estimated_angle:.2f}° por debajo del umbral de aplicación {deskew_app_thresh}°. No se aplica corrección de rotación.") #

        # --- 4. Reducción de Ruido (Denoise) ---
        denoise_strength_val = self._determine_denoise_strength(quality_assessment_metrics, active_config)
        denoised_image = cleaners.detail_preserving_denoise(gray_image_deskewed, denoise_strength_val)
        logger.debug(f"Denoising aplicado con strength: {denoise_strength_val}.")

        # --- 5. Mejora de Contraste (CLAHE) ---
        clahe_clip = float(self._get_param_from_config(active_config, 'contrast_enhancement.clahe_clip_limit', 2.0))
        clahe_grid = self._determine_clahe_tile_grid_size(quality_assessment_metrics, active_config)
        enhanced_image = enhancers.conservative_contrast_enhancement(denoised_image, clahe_clip, clahe_grid)
        logger.debug(f"Mejora de contraste (CLAHE) aplicada con clip: {clahe_clip}, grid: {clahe_grid}.")

        # --- 6. Binarización ---
        bin_block_size = self._determine_binarization_block_size(quality_assessment_metrics, active_config)
        bin_c_val = int(self._get_param_from_config(active_config, 'binarization.adaptive_c_value', 5))
        binary_image_for_ocr = cleaners.high_res_binarization(enhanced_image, bin_block_size, bin_c_val)
        if binary_image_for_ocr is None:
            logger.error("Fallo en la binarización después de mejoras.")
            elapsed_time = time.perf_counter() - stage_start_time
            return None, elapsed_time
        logger.info(f"Binarización adaptativa aplicada con block_size: {bin_block_size}, C: {bin_c_val}.")

        results_dict = {
            "binary_image_for_ocr": binary_image_for_ocr,
            "gray_image_deskewed": gray_image_deskewed, 
            "preprocessing_parameters_used": { 
                "estimated_skew_before_correction": estimated_angle,
                "deskew_application_threshold": deskew_app_thresh,
                "angle_corrected_by": actual_angle_corrected,
                "denoise_strength": denoise_strength_val,
                "clahe_clip_limit": clahe_clip,
                "clahe_tile_grid_size": clahe_grid,
                "binarization_block_size": bin_block_size,
                "binarization_c_value": bin_c_val,
            }
        }
        elapsed_time = time.perf_counter() - stage_start_time # Calcular tiempo total del pipeline
        return results_dict, elapsed_time 
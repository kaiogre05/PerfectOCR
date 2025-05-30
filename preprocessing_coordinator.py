# PerfectOCR/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import os
import logging
import time
from typing import Any, Optional, Dict, Tuple
from core.preprocessing.image_corrector import ImageCorrector
from core.preprocessing import toolbox # Importa el toolbox directamente
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PreprocessingCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.corrector = ImageCorrector()
        self.binarization_rules = config.get('quality_assessment_rules', {}).get('binarization', {})

    def apply_preprocessing_pipelines(
        self,
        image_array: np.ndarray,
        correction_plans: Dict[str, Any],
        image_path_for_log: str
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Genera imágenes optimizadas específicamente para cada motor OCR
        - Tesseract: imagen binarizada escalada para mejor reconocimiento de palabras
        - PaddleOCR: imagen en escala de grises con CLAHE
        """
        start_time = time.perf_counter()
        
        if not correction_plans:
            logger.warning("El plan de corrección está vacío. No se aplicará preprocesamiento.")
            return None, time.perf_counter() - start_time

        # 1. Preparar imagen base
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
        
        # 2. Procesar específicamente para cada motor OCR
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            for engine, plan in correction_plans.items():
                if not plan: continue
                
                # Asegurar estructura correcta del plan
                if isinstance(plan, (int, float, np.number)):
                    plan = {
                        'deskew': {'angle': float(plan)},
                        'denoise': {'strength': 5},
                        'contrast': {'clahe_params': {'clip_limit': 1.5, 'grid_size': (8, 8)}}
                    }
                
                if engine == 'tesseract':
                    # Procesamiento especializado para Tesseract: binarización + escalado
                    futures[engine] = executor.submit(
                        self._process_for_tesseract,
                        gray_image.copy(),
                        plan
                    )
                else:
                    # Procesamiento estándar para PaddleOCR: escala de grises con CLAHE
                    futures[engine] = executor.submit(
                        self.corrector.apply_grayscale_corrections,
                        gray_image.copy(),
                        plan
                    )
            
            # Recolectar resultados
            ocr_grayscale_images = {
                engine: future.result()
                for engine, future in futures.items()
            }
        
        results_dict = {
            "ocr_images": ocr_grayscale_images,
            "preprocessing_parameters_used": correction_plans
        }
        
        elapsed_time = time.perf_counter() - start_time
        return results_dict, elapsed_time

    def _process_for_tesseract(self, gray_image: np.ndarray, correction_plan: Dict[str, Any]) -> np.ndarray:
        """
        Procesamiento especializado para Tesseract:
        1. Aplicar correcciones básicas (deskew, denoise, contrast)
        2. Escalar imagen a 300 DPI
        3. Binarizar para texto negro sobre fondo blanco
        """
        # 1. Aplicar correcciones estándar
        corrected_image = self.corrector.apply_grayscale_corrections(gray_image, correction_plan)
        
        # 2. Escalado para aproximar 300 DPI
        scale_factor = 1.5
        upscaled = cv2.resize(corrected_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 3. Binarización adaptativa optimizada para texto
        block_size = 31  # Tamaño de bloque para binarización adaptativa
        c_value = 7      # Constante sustraída de la media
        binary_image = toolbox.apply_binarization(upscaled, block_size, c_value)
        
        # 4. Invertir para asegurar texto negro sobre fondo blanco
        final_image = cv2.bitwise_not(binary_image)
        
        return final_image
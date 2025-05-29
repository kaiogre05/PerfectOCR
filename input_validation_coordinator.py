# PerfectOCR/coordinators/input_validation_coordinator.py
import os
import logging
import cv2
import time
from typing import Dict, Any, Tuple, List, Optional
from core.input_validation.quality_evaluator import ImageQualityEvaluator

logger = logging.getLogger(__name__)

class InputValidationCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.config = config
        self.project_root = project_root
        self.quality_evaluator = ImageQualityEvaluator(config = self.config)
        logger.info("InputValidationCoordinator inicializado.")

    def validate_and_assess_image(self, input_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[cv2.typing.MatLike], float]:
        """
        Carga la imagen UNA VEZ y luego llama al evaluador para obtener métricas y observaciones.
        Devuelve: (métricas de calidad, observaciones de calidad, array de imagen cargado, tiempo de ejecución)
        """
        stage_start_time = time.perf_counter()
        logger.info(f"InputValidationCoordinator: Cargando y evaluando calidad para {os.path.basename(input_path)}")
        image_loaded_array: Optional[cv2.typing.MatLike] = None
        quality_metrics: Optional[Dict[str, Any]] = None
        quality_observations: Optional[List[str]] = None

        try:
            image_loaded_array = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if image_loaded_array is None:
                logger.error(f"No se pudo cargar la imagen (o está vacía) desde: {input_path}")
                quality_metrics = {'error': f"Failed to load image: {input_path}", 'input_path_reference': input_path}
                quality_observations = ["error_loading_image_file"]
                elapsed_time = time.perf_counter() - stage_start_time
                return quality_metrics, quality_observations, None, elapsed_time
        except Exception as e:
            logger.error(f"Excepción al cargar la imagen {input_path} en InputValidationCoordinator: {e}", exc_info=True)
            quality_metrics = {'error': f"Exception loading image {input_path}: {e}", 'input_path_reference': input_path}
            quality_observations = ["error_exception_on_load"]
            elapsed_time = time.perf_counter() - stage_start_time # Calculate time before returning
            return quality_metrics, quality_observations, None, elapsed_time

        quality_metrics, quality_observations = self.quality_evaluator.evaluate_image_metrics(
            image_array=image_loaded_array,
            input_path_for_log=input_path
        )
        
        elapsed_time = time.perf_counter() - stage_start_time # Calculate final time
        return quality_metrics, quality_observations, image_loaded_array, elapsed_time
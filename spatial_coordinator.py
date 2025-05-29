# PerfectOCR/coordinators/spatial_analyzer_coordinator.py
import logging
import numpy as np
from typing import Dict, Any, Optional
from core.spatial_analysis.density_calculator import calculate_density_map

logger = logging.getLogger(__name__)

class SpatialAnalyzerCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.specific_config = config # CORRECCIÓN: Usar directamente el 'config' que ya es la sección correcta.
        self.project_root = project_root
        # El log original aquí es demasiado verboso para INFO, se deja como estaba implícitamente en DEBUG o se elimina.
        logger.info("SpatialAnalyzerCoordinator (Enfoque Puro en Densidad) inicializado.")
        default_window_size_from_yaml = 15
        # CORRECCIÓN: Leer el parámetro desde self.specific_config
        self.density_map_window_size = int(self.specific_config.get('density_map_window_size', default_window_size_from_yaml))

        if self.density_map_window_size <= 0 or self.density_map_window_size % 2 == 0:
            logger.warning(
                f"density_map_window_size ({self.density_map_window_size}) en config no es válido "
                f"(debe ser impar y >0). Usando valor por defecto: {default_window_size_from_yaml}."
            )
            self.density_map_window_size = default_window_size_from_yaml
        
        logger.debug(f"Configuración de SpatialAnalyzer: Ventana de Densidad = {self.density_map_window_size}")

    def analyze_image(self, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Realiza el análisis espacial (solo mapa de densidad) sobre la imagen binarizada.

        Args:
            binary_image (np.ndarray): La imagen binarizada (se espera uint8, ej. texto=0, fondo=255
                                           o texto=255, fondo=0).

        Returns:
            Dict[str, Any]: Un diccionario con el mapa de densidad y dimensiones.
                           El mapa de densidad puede ser None si ocurren errores.
        """
        if binary_image is None:
            logger.error("analyze_image (Spatial) recibió una imagen None.")
            return {
                "density_map": None,
                "image_dims": None,
                "error": "Input image was None"
            }

        h, w = binary_image.shape[:2]
        logger.info(f"SpatialAnalyzer iniciando análisis de densidad para imagen de {w}x{h}.")

        mean_pixel_value = np.mean(binary_image)
        if mean_pixel_value > 127: 
            binary_image_normalized_text_as_1 = (binary_image == 0).astype(np.float32)
        else: 
            binary_image_normalized_text_as_1 = (binary_image == 255).astype(np.float32)
        
        density_map_result = calculate_density_map(
            binary_image_normalized_text_as_1, 
            self.density_map_window_size
        )

        logger.info("Análisis de densidad (solo mapa) completado por SpatialAnalyzerCoordinator.")
        return {
            "density_map": density_map_result,
            "image_dims": {"width": w, "height": h}
        }

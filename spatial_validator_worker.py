# PerfectOCR/core/table_extractor/spatial_validator_worker.py
import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SpatialValidatorWorker:
    def __init__(self, config: Dict):
        self.config = config
        self.min_word_density_score_for_consideration = float(
            config.get('min_word_density_score_for_consideration', {})
        )
        logger.info(
            f"SpatialValidatorWorker inicializado. "
            f"Umbral densidad palabra: {self.min_word_density_score_for_consideration}"
        )

    def calculate_word_density_score(
        self,
        word_original_geometry: Dict[str, Any],
        density_map: Optional[np.ndarray]
    ) -> float:
        """
        Calcula la densidad promedio bajo el polígono/bbox de la palabra usando el density_map.
        word_original_geometry es el contenido de 'original_geometry_data' de una palabra.
        Devuelve un score entre 0.0 y 1.0.
        """
        if density_map is None:
            return 0.5

        poly_coords_list = word_original_geometry.get('polygon_coords')
        bbox_list = word_original_geometry.get('bbox')

        xmin_w, ymin_w, xmax_w, ymax_w = -1.0, -1.0, -1.0, -1.0

        if isinstance(poly_coords_list, list) and len(poly_coords_list) >= 3:
            try:
                np_poly_pts = np.array([[float(p[0]), float(p[1])] for p in poly_coords_list])
                min_coords = np_poly_pts.min(axis=0)
                max_coords = np_poly_pts.max(axis=0)
                xmin_w, ymin_w = min_coords[0], min_coords[1]
                xmax_w, ymax_w = max_coords[0], max_coords[1]
            except (TypeError, ValueError, IndexError) as e:
                logger.debug(f"Error procesando poly_coords para density score: {e}")
                if not (isinstance(bbox_list, list) and len(bbox_list) == 4):
                    return 0.0
        
        if (xmin_w == -1) and isinstance(bbox_list, list) and len(bbox_list) == 4:
            try:
                xmin_w, ymin_w, xmax_w, ymax_w = float(bbox_list[0]), float(bbox_list[1]), float(bbox_list[2]), float(bbox_list[3])
            except (TypeError, ValueError) as e:
                logger.debug(f"Error procesando bbox para density score: {e}")
                return 0.0

        if xmin_w == -1 or xmax_w <= xmin_w or ymax_w <= ymin_w:
            logger.debug(f"Geometría inválida para calcular density score.")
            return 0.0

        h_map, w_map = density_map.shape
        ixmin = max(0, int(round(xmin_w)))
        iymin = max(0, int(round(ymin_w)))
        ixmax = min(w_map, int(round(xmax_w))) 
        iymax = min(h_map, int(round(ymax_w)))

        if ixmin >= ixmax or iymin >= iymax:
            return 0.0

        density_patch = density_map[iymin:iymax, ixmin:ixmax]
        if density_patch.size == 0:
            return 0.0

        avg_density = np.mean(density_patch)
        return float(avg_density) if not np.isnan(avg_density) else 0.0

    def calculate_cell_region_density(
        self,
        cell_bounds: Dict[str, float],
        density_map: Optional[np.ndarray]
    ) -> float:
        """
        Calcula la densidad promedio de una región de celda definida por sus límites.
        """
        if density_map is None:
            return 0.5

        xmin_c, ymin_c = cell_bounds.get('xmin', -1.0), cell_bounds.get('ymin', -1.0)
        xmax_c, ymax_c = cell_bounds.get('xmax', -1.0), cell_bounds.get('ymax', -1.0)

        if xmin_c == -1 or xmax_c <= xmin_c or ymax_c <= ymin_c:
            logger.debug(f"Límites de celda inválidos para calcular densidad de región: {cell_bounds}")
            return 0.0
            
        h_map, w_map = density_map.shape
        ixmin = max(0, int(round(xmin_c)))
        iymin = max(0, int(round(ymin_c)))
        ixmax = min(w_map, int(round(xmax_c)))
        iymax = min(h_map, int(round(ymax_c)))

        if ixmin >= ixmax or iymin >= iymax:
            return 0.0
        
        density_patch = density_map[iymin:iymax, ixmin:ixmax]
        if density_patch.size == 0:
            return 0.0
            
        avg_density = np.mean(density_patch)
        return float(avg_density) if not np.isnan(avg_density) else 0.0
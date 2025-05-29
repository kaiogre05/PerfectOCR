# PerfectOCR/core/table_extractor/table_boundary_detector.py
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .header_detector import HeaderDetector
from utils.geometric import get_polygon_bounds

logger = logging.getLogger(__name__)

class TableBoundaryDetector:
    def __init__(self, config: Dict, project_root: str):
        self.config = config
        self.project_root = project_root
        self.page_dimensions: Dict[str, Any] = {}
        self.header_detector_config = self.config.get('header_detector_config', {})
        self.spatial_table_end_config = self.config.get('spatial_table_end_detection', {})
        self.table_end_keywords = [k.upper() for k in self.header_detector_config.get('table_end_keywords', [])]
        self.header_keywords_list = self.header_detector_config.get('table_header_keywords_list', []) 
        self.header_keywords_for_detector = self.header_detector_config.get('table_header_keywords_list', [])
        self.header_detector_instance: Optional[HeaderDetector] = None

    def set_page_dimensions(self, page_dimensions: Dict[str, Any]):
        if page_dimensions and page_dimensions.get('width') and page_dimensions.get('height'):
            self.page_dimensions = page_dimensions
            logger.debug(f"TableBoundaryDetector: Dimensiones de página establecidas a {self.page_dimensions}")
            if self.header_detector_instance:
                self.header_detector_instance.set_page_dimensions(self.page_dimensions)
        else:
            logger.warning(f"TableBoundaryDetector: Intento de establecer dimensiones de página inválidas: {page_dimensions}")

    def _find_table_end_y_spatial(self, 
                                density_map: np.ndarray, 
                                y_start_search: float, 
                                page_height: int) -> float:
        if density_map is None:
            logger.warning("BoundaryDetector: Mapa de densidad no disponible para _find_table_end_y_spatial. Usando final de página.")
            return float(page_height)

        cfg = self.spatial_table_end_config
        density_drop_threshold_ratio = cfg.get('density_drop_threshold_ratio', {}) 
        min_low_density_rows_to_confirm_end = cfg.get('min_low_density_rows_to_confirm_end', {}) 
        smoothing_window_size = cfg.get('smoothing_window_size', {})
        if smoothing_window_size % 2 == 0 : smoothing_window_size +=1 
        ignore_bottom_page_ratio = cfg.get('ignore_bottom_page_ratio', {})

        iy_start_search = int(round(y_start_search))
        if iy_start_search >= page_height: return float(page_height)

        try:
            horizontal_profile = np.sum(density_map[iy_start_search:, :], axis=1)
        except IndexError:
            logger.warning(f"BoundaryDetector: Error al calcular perfil horizontal (y_start_search={iy_start_search}, shape={density_map.shape}). Usando final de página.")
            return float(page_height)
        if horizontal_profile.size == 0: return float(page_height)
        if smoothing_window_size > 1 and len(horizontal_profile) >= smoothing_window_size:
            smoothed_profile = np.convolve(horizontal_profile, np.ones(smoothing_window_size)/smoothing_window_size, mode='valid')
        else:
            smoothed_profile = horizontal_profile
        if smoothed_profile.size == 0: return float(page_height)
        
        initial_data_region_len = min(len(smoothed_profile), max(min_low_density_rows_to_confirm_end * 2, int(len(smoothed_profile) * 0.2)))
        reference_density_calc_profile = smoothed_profile[:initial_data_region_len] if initial_data_region_len > 0 else horizontal_profile[:min(len(horizontal_profile), min_low_density_rows_to_confirm_end * 3)]
        
        if reference_density_calc_profile.size > 0:
            reference_density = np.percentile(reference_density_calc_profile, 80) if initial_data_region_len > 0 else np.max(reference_density_calc_profile)
        else:
            reference_density = 0.0
        if reference_density < 1e-3: return float(page_height)
        low_density_threshold = reference_density * density_drop_threshold_ratio
        consecutive_low_density_rows = 0
        limit_y_search = page_height * (1 - ignore_bottom_page_ratio)

        for i, density_value in enumerate(smoothed_profile):
            current_y_on_page = iy_start_search + i + (smoothing_window_size // 2) 
            if current_y_on_page > limit_y_search: break 
            if density_value < low_density_threshold:
                consecutive_low_density_rows += 1
            else:
                consecutive_low_density_rows = 0 
            if consecutive_low_density_rows >= min_low_density_rows_to_confirm_end:
                table_end_y = iy_start_search + (i - min_low_density_rows_to_confirm_end + 1) + (smoothing_window_size // 2)
                logger.info(f"BoundaryDetector: Final de tabla detectado espacialmente en Y={table_end_y}")
                return float(max(y_start_search, table_end_y))
        
        return float(page_height)

    def detect_boundaries(self, 
                        lines_for_search: List[Dict[str, Any]], 
                        density_map: Optional[np.ndarray],
                        page_dimensions: Dict[str, Any]) -> Dict[str, Any]:
        
        self.set_page_dimensions(page_dimensions)
        
        if not self.page_dimensions or not self.page_dimensions.get('width'):
            logger.error("TableBoundaryDetector: No se puede proceder sin dimensiones de página válidas.")
            return {"error": "Invalid page dimensions"}

        if not self.header_detector_instance:
            self.header_detector_instance = HeaderDetector(
                config=self.header_detector_config,
                header_keywords_list=self.header_keywords_for_detector, # <--- CORREGIR ESTA LÍNEA
                page_dimensions=self.page_dimensions
            )

        header_words, y_min_h_band, y_max_h_band = self.header_detector_instance.identify_header_band_and_words(
            formed_lines = lines_for_search
        )

        if not header_words or y_max_h_band is None:
            logger.warning("TableBoundaryDetector: No se pudo identificar la banda de encabezados.")
            return {"header_words": None, "y_max_header_band": None, "y_min_table_end": None, "error": "Header band not identified."}

        page_height_val = self.page_dimensions.get('height')
        
        # Eliminé la función _find_table_end_y_coordinate ya que no estaba definida en el código original
        y_min_table_end_by_keyword = None  # Esto debería ser reemplazado por la lógica adecuada
        
        y_min_table_end_spatial = float(page_height_val) if page_height_val is not None else float('inf')
        if density_map is not None and page_height_val is not None and page_height_val > 0:
            y_min_table_end_spatial = self._find_table_end_y_spatial(
                density_map, y_max_h_band, int(page_height_val)
            )
        
        y_min_table_end = y_min_table_end_by_keyword 
        default_word_h_heuristic = self.header_detector_config.get('default_line_height_for_gap', {})

        if y_min_table_end_spatial is not None and y_max_h_band is not None:
            if y_min_table_end_spatial > y_max_h_band + default_word_h_heuristic: 
                if y_min_table_end_by_keyword is None or y_min_table_end_spatial < y_min_table_end_by_keyword:
                    y_min_table_end = y_min_table_end_spatial
        
        if y_min_table_end is None and page_height_val is not None:
            y_min_table_end = float(page_height_val)
        elif y_min_table_end is None:
            logger.error("TableBoundaryDetector: No se pudo determinar y_min_table_end y page_height es None.")
            return {"header_words": header_words, "y_max_header_band": y_max_h_band, "y_min_table_end": None, "error": "Could not determine table end."}

        logger.info(f"TableBoundaryDetector: Final de tabla efectivo determinado en Y={y_min_table_end:.2f} (Keyword: {y_min_table_end_by_keyword}, Spatial: {y_min_table_end_spatial})")
        
        return {
            "header_words": header_words,
            "y_max_header_band": y_max_h_band,
            "y_min_table_end": y_min_table_end
        }
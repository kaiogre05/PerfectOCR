# PerfectOCR/core/table_extractor/table_data_preparer.py
import logging
import numpy as np
from typing import Dict, List, Any, Optional

from utils.geometric import get_polygon_bounds, get_shapely_polygon
from .spatial_validator_worker import SpatialValidatorWorker

logger = logging.getLogger(__name__)

class TableDataPreparer:
    def __init__(self, data_filter_config: Dict, spatial_validator_worker: SpatialValidatorWorker):
        self.data_filter_cfg = data_filter_config
        self.spatial_validator = spatial_validator_worker # Usar la instancia pasada
        
        # Cargar umbrales del data_filter_config
        self.min_line_avg_conf = self.data_filter_cfg.get('min_line_avg_confidence_for_table', {})
        self.min_word_conf = self.data_filter_cfg.get('min_word_confidence_for_table', {})
        # Usar el valor del YAML; SpatialValidatorWorker ya tiene su propia config para min_word_density_score_for_consideration
        self.min_word_density_for_table = self.data_filter_cfg.get('min_word_density_score_for_table', {}) 
        self.filter_short_non_alphanum = self.data_filter_cfg.get('filter_short_non_alphanum_lines', {})
        self.short_line_max_chars = self.data_filter_cfg.get('short_line_max_chars_for_filter', {})
        self.noise_max_ar = self.data_filter_cfg.get('noise_line_max_aspect_ratio', {})
        self.noise_min_ar = self.data_filter_cfg.get('noise_line_min_aspect_ratio', {})
        self.noise_min_words_extreme_ar = self.data_filter_cfg.get('noise_line_min_words_for_extreme_aspect_ratio', {})
        self.noise_min_conf_extreme_ar = self.data_filter_cfg.get('noise_line_min_conf_for_extreme_aspect_ratio', {})
        logger.info("TableDataPreparer inicializado.")

    def _recalculate_word_geometry(self, polygon_coords: Optional[List[List[float]]]) -> Optional[Dict[str, float]]:
        if not polygon_coords or not isinstance(polygon_coords, list) or len(polygon_coords) < 3:
            return None
        try:
            float_poly_coords = [[float(p[0]), float(p[1])] for p in polygon_coords]
            s_poly = get_shapely_polygon(float_poly_coords)
            if not s_poly or s_poly.is_empty or not s_poly.is_valid:
                return None

            min_x, min_y, max_x, max_y = s_poly.bounds
            height = max_y - min_y
            width = max_x - min_x
            if height <= 0.1 or width <= 0.1: return None

            return {
                'cx': s_poly.centroid.x, 'cy': s_poly.centroid.y,
                'xmin': min_x, 'ymin': min_y, 'xmax': max_x, 'ymax': max_y,
                'height': height, 'width': width,
                'polygon_coords': float_poly_coords
            }
        except Exception as e:
            logger.warning(f"Error recalculando geometría para palabra con coords {polygon_coords}: {e}", exc_info=False)
            return None

    def prepare(self, 
                reconstructed_fused_lines: List[Dict[str, Any]], 
                density_map: Optional[np.ndarray],
                page_dimensions: Dict[str, Any] 
               ) -> Dict[str, List[Any]]:
        
        prepared_lines_for_boundary_detection: List[Dict[str, Any]] = []
        all_valid_words_for_table_processing: List[Dict[str, Any]] = []
        element_id_counter = 0 # Para IDs únicos si es necesario

        for line_idx, fused_line_item in enumerate(reconstructed_fused_lines):
            line_text_raw = fused_line_item.get('text_raw', '').strip()
            line_avg_conf = fused_line_item.get('avg_constituent_confidence', 0.0)
            
            if line_avg_conf < self.min_line_avg_conf:
                logger.debug(f"[Preparer] Filtrando línea (conf. línea {line_avg_conf:.2f} < {self.min_line_avg_conf:.2f}): '{line_text_raw[:50]}...'")
                continue
            
            if self.filter_short_non_alphanum and \
               len(line_text_raw.replace(" ","")) < self.short_line_max_chars and \
               not any(char.isalnum() for char in line_text_raw):
                logger.debug(f"[Preparer] Filtrando línea (corta no alfanumérica): '{line_text_raw[:50]}...'")
                continue
                
            line_poly_coords = fused_line_item.get('polygon_line_bbox')
            if line_poly_coords:
                try:
                    line_bounds = get_polygon_bounds(line_poly_coords)
                    if line_bounds and line_bounds[2] > line_bounds[0] and line_bounds[3] > line_bounds[1]: # xmax > xmin, ymax > ymin
                        line_width = line_bounds[2] - line_bounds[0]
                        line_height = line_bounds[3] - line_bounds[1]
                        if line_width > 1 and line_height > 1: # Evitar división por cero y dimensiones mínimas
                            aspect_ratio = line_width / line_height
                            if aspect_ratio > self.noise_max_ar or aspect_ratio < self.noise_min_ar:
                                if len(line_text_raw.split()) < self.noise_min_words_extreme_ar and \
                                   line_avg_conf < self.noise_min_conf_extreme_ar:
                                    logger.debug(f"[Preparer] Filtrando línea (aspecto extremo {aspect_ratio:.2f} y bajo contenido/conf): '{line_text_raw[:50]}...'")
                                    continue
                except Exception as e_geom:
                    logger.warning(f"[Preparer] Error procesando geometría de línea {fused_line_item.get('line_id','N/A')} para filtro aspect ratio: {e_geom}")

            constituent_elements_for_line_output = []
            valid_words_in_line_count = 0
            line_text_raw_from_constituents = ""
            
            for element_data_in_fused_line in fused_line_item.get('constituent_elements_ocr_data', []):
                current_element_proc = element_data_in_fused_line.copy()
                geom_recalc = self._recalculate_word_geometry(current_element_proc.get('polygon_coords'))
                if geom_recalc:
                    current_element_proc.update(geom_recalc)
                else: # Si la geometría es inválida después del recalculo
                    logger.warning(f"[Preparer] Geometría inválida para palabra '{current_element_proc.get('text')}' en línea ID {fused_line_item.get('line_id')}. Omitiendo palabra.")
                    continue # Saltar esta palabra
                
                if not all(k in current_element_proc for k in ['cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width']):
                    logger.warning(f"[Preparer] Omitiendo palabra constituyente '{current_element_proc.get('text')}' por falta de geometría completa en línea ID {fused_line_item.get('line_id')}")
                    continue
                
                # Calcular densidad usando el worker de validación espacial
                word_conf = current_element_proc.get('confidence', 0.0)
                word_density_score = current_element_proc.get('density_score', 0.0) # Asumiendo que 'density_score' se añade a current_element_proc

                # Aplicar filtros a nivel de palabra
                if word_conf < self.min_word_conf or word_density_score < self.min_word_density_for_table:
                    logger.debug(f"[Preparer] Filtrando palabra DENTRO de línea (conf {word_conf:.2f} < {self.min_word_conf:.2f} o dens {word_density_score:.2f} < {self.min_word_density_for_table:.2f}): '{current_element_proc.get('text')}'")
                    continue

                constituent_elements_for_line_output.append(current_element_proc)
                word_for_global_list = current_element_proc.copy()
                word_for_global_list['original_id_for_table'] = word_for_global_list.get('internal_id', f"elem_prep_{element_id_counter}")
                all_valid_words_for_table_processing.append(word_for_global_list)
                element_id_counter +=1
                valid_words_in_line_count +=1

            if (not constituent_elements_for_line_output or valid_words_in_line_count == 0) and \
               fused_line_item.get('avg_constituent_confidence', 0.0) >= self.min_line_avg_conf and \
               fused_line_item.get('fusion_source', '').startswith('paddle_text_preferred'): # Solo si Paddle fue preferido

                line_text_raw_override = fused_line_item.get('text_raw','').strip()
                logger.debug(f"[Preparer] Línea ID {fused_line_item.get('line_id')} tenía todas sus palabras filtradas, pero la línea fusionada (probablemente de Paddle) tiene alta confianza ({fused_line_item.get('avg_constituent_confidence', 0.0):.2f}). "
                            f"Creando pseudo-palabras a partir del texto de línea: '{line_text_raw_override[:50]}'")
                
                # Dividir el texto de la línea de Paddle en palabras y crear pseudo-constituyentes
                paddle_words_split = line_text_raw_override.split()
                line_poly_bbox = fused_line_item.get('polygon_line_bbox') # Geometría de toda la línea
                
                if paddle_words_split and line_poly_bbox:
                    line_bounds_for_split = get_polygon_bounds(line_poly_bbox)
                    if line_bounds_for_split and line_bounds_for_split[2] > line_bounds_for_split[0]: # xmax > xmin
                        line_xmin, line_ymin, line_xmax, line_ymax = line_bounds_for_split
                        approx_word_width = (line_xmax - line_xmin) / len(paddle_words_split) if len(paddle_words_split) > 0 else (line_xmax - line_xmin)

                        for i, p_word_text in enumerate(paddle_words_split):
                            word_xmin_approx = line_xmin + (i * approx_word_width)
                            word_xmax_approx = line_xmin + ((i + 1) * approx_word_width)
                            
                            pseudo_poly_coords = [[word_xmin_approx, line_ymin], [word_xmax_approx, line_ymin], 
                                                  [word_xmax_approx, line_ymax], [word_xmin_approx, line_ymax]]
                            geom_recalc_pseudo = self._recalculate_word_geometry(pseudo_poly_coords)

                            if geom_recalc_pseudo:
                                pseudo_word = {
                                    'text': p_word_text,
                                    'confidence': fused_line_item.get('avg_constituent_confidence', 0.0), # Usar la confianza alta de la línea
                                    'density_score': 0.5, # Puntuación neutral o recalcular si es posible
                                    'internal_id': f"{fused_line_item.get('line_id')}_pseudo_word_{i}",
                                    'original_id_for_table': f"{fused_line_item.get('line_id')}_pseudo_word_{i}",
                                    'source_ocr': 'paddle_override_pseudo' # Indicar origen
                                }
                                pseudo_word.update(geom_recalc_pseudo)
                                constituent_elements_for_line_output.append(pseudo_word)
                                # También añadir a la lista global de palabras si es necesario para RowDefiner/CellAssigner
                                all_valid_words_for_table_processing.append(pseudo_word.copy()) 
                                valid_words_in_line_count +=1
                            else:
                                logger.warning(f"No se pudo recalcular geometría para pseudo-palabra '{p_word_text}' en línea {fused_line_item.get('line_id')}")
                    else: # Si no se pudo dividir en palabras o no hay geometría de línea
                        logger.warning(f"No se pudieron crear pseudo-palabras para la línea {fused_line_item.get('line_id')} a pesar de la alta confianza de línea.")


            # Verificar de nuevo si la línea está vacía
            if not constituent_elements_for_line_output or valid_words_in_line_count == 0:
                logger.debug(f"[Preparer] Línea ID {fused_line_item.get('line_id')} vacía tras filtrar palabras (y posible intento de rescate). Omitiendo para detección de encabezado.")
                continue
            
            # Reconstruir texto de línea y confianza promedio si se filtraron palabras internas
            line_text_for_header_detector = " ".join([el.get('text', '') for el in constituent_elements_for_line_output]).strip()
            confidences_for_header_line = [el.get('confidence', 0.0) for el in constituent_elements_for_line_output if el.get('confidence') is not None]
            line_avg_conf_for_header_detector = round(float(np.mean(confidences_for_header_line)), 2) if confidences_for_header_line else 0.0

            adapted_line_for_hd = {
                'line_id': fused_line_item.get('line_id', f"prep_line_{line_idx}"),
                'polygon_final': fused_line_item.get('polygon_line_bbox'),
                'constituent_elements': constituent_elements_for_line_output,
                'structural_confidence_pct': line_avg_conf_for_header_detector, # Usar la confianza recalculada
                'text_raw': line_text_for_header_detector, # Usar el texto recalculado
                'source_info': fused_line_item.get('fusion_source', 'unknown') # Mantener la fuente original de la línea
            }
            prepared_lines_for_boundary_detection.append(adapted_line_for_hd)

        logger.info(f"[Preparer] Datos preparados: {len(prepared_lines_for_boundary_detection)} líneas candidatas para límites/encabezado, "
                    f"{len(all_valid_words_for_table_processing)} palabras válidas totales para celdas.")
        
        return {
            "header_candidate_lines": prepared_lines_for_boundary_detection,
            "all_valid_words_for_table": all_valid_words_for_table_processing # Esta es la lista de palabras individuales
        }
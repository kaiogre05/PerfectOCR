# PerfectOCR/core/table_extractor/table_cell_assigner.py
import logging
import math
from typing import List, Dict, Optional, Any
from utils.geometric import calculate_iou

logger = logging.getLogger(__name__)

class TableCellAssigner:
    def __init__(self, config: Dict):
        self.config = config
        self.cell_cosine_strong_threshold = float(config.get('cell_cosine_strong_threshold', {}))
        self.fallback_cell_iou_threshold = float(config.get('fallback_cell_iou_threshold', {}))
        # El umbral de densidad de palabra ya no se maneja aquí directamente
        logger.debug(
            f"TableCellAssigner (simplificado) inicializado con umbral coseno: {self.cell_cosine_strong_threshold}, "
            f"umbral IoU fallback: {self.fallback_cell_iou_threshold}"
        )

    def _get_word_polygon_for_iou(self, word_dict_vectorized: Dict[str, Any]) -> Optional[List[List[float]]]:
        # ... (código existente sin cambios) ...
        original_geom = word_dict_vectorized.get('original_geometry_data', {})

        poly_coords_raw = original_geom.get('polygon_coords')
        if poly_coords_raw and isinstance(poly_coords_raw, list) and len(poly_coords_raw) >= 3:
            try:
                valid_poly_coords = [[float(p[0]), float(p[1])] for p in poly_coords_raw
                                    if isinstance(p, (list, tuple)) and len(p) == 2]
                if len(valid_poly_coords) >= 3:
                    return valid_poly_coords
            except (TypeError, ValueError, IndexError) as e:
                logger.debug(f"Error convirtiendo polygon_coords a float en _get_word_polygon_for_iou para '{word_dict_vectorized.get('text', 'N/A')}': {e}")

        bbox_raw = original_geom.get('bbox')
        if bbox_raw and isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            try:
                if all(isinstance(coord, (int, float)) for coord in bbox_raw):
                    xmin, ymin, xmax, ymax = float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3])
                    if xmax > xmin and ymax > ymin:
                        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            except (TypeError, ValueError) as e:
                 logger.debug(f"Error convirtiendo bbox a float en _get_word_polygon_for_iou para '{word_dict_vectorized.get('text', 'N/A')}': {e}")
        
        logger.debug(f"Palabra '{word_dict_vectorized.get('text', 'N/A')}' sin 'polygon_coords' o 'bbox' válidos en 'original_geometry_data' para cálculo de IoU. Geometría original: {original_geom}")
        return None


    def assign_words_to_cells(self,
                              data_rows_with_words: Optional[List[Dict[str, Any]]],
                              defined_columns: Optional[List[Dict[str, Any]]]
                             ) -> List[List[Dict[str, Any]]]:
        # El parámetro density_map ya no es necesario aquí si las palabras son pre-filtradas/enriquecidas
        # por TableAndFieldCoordinator

        current_data_rows = data_rows_with_words if data_rows_with_words is not None else []
        current_defined_columns = defined_columns if defined_columns is not None else []

        num_rows_m = len(current_data_rows)
        num_cols_n = len(current_defined_columns)

        if num_rows_m == 0 or num_cols_n == 0:
            logger.warning("TableCellAssigner: No hay filas o columnas válidas para la asignación de celdas.")
            return []

        # Inicializar la matriz de celdas
        # La estructura 'cell_metrics' será poblada por TableAndFieldCoordinator
        table_matrix_cells = [[{
            "text": "", "words": [], "assignment_confidence": "none",
            "cell_metrics": {} # Inicializar vacío, se llenará después
        } for _ in range(num_cols_n)] for _ in range(num_rows_m)]
        processed_word_original_ids = set()

        # Iteración 1: Asignación Primaria (Coseno)
        # Asumimos que las palabras en data_rows_with_words ya han sido validadas por densidad
        # por el TableAndFieldCoordinator si se decidió pre-filtrar.
        for r_idx, row_definition in enumerate(current_data_rows):
            for word_dict_vec in row_definition.get('words_in_row', []):
                word_id_key = word_dict_vec.get('original_id_for_table', word_dict_vec.get('internal_id', str(id(word_dict_vec))))

                if word_id_key in processed_word_original_ids:
                    continue
                
                # La validación de densidad ya no se hace aquí. Se asume que `word_dict_vec`
                # es una palabra que vale la pena procesar.

                vector_rep = word_dict_vec.get('vector_representation')
                if not vector_rep or not isinstance(vector_rep, list) or len(vector_rep) < 2 or \
                   not (isinstance(vector_rep[0], (int,float)) and isinstance(vector_rep[1], (int,float))):
                    logger.debug(f"Omitiendo palabra '{word_dict_vec.get('text', 'N/A')}' sin 'vector_representation' numérico y válido.")
                    continue
                xc, yc = float(vector_rep[0]), float(vector_rep[1])

                best_k_cos = -1
                max_cos_score = -float('inf')
                for k_idx, col_info in enumerate(current_defined_columns):
                    X_k_header_center = col_info.get('X_k', xc)
                    Y_H_k_header_center = col_info.get('Y_H_k', yc - 1)
                    vec_x_val = xc - X_k_header_center
                    vec_y_val = yc - Y_H_k_header_center
                    magnitude = math.sqrt(vec_x_val**2 + vec_y_val**2)
                    current_cos_score = (vec_y_val / magnitude) if magnitude > 1e-6 else (1.0 if vec_y_val >= 0 else -1.0)
                    if current_cos_score > max_cos_score:
                        max_cos_score = current_cos_score
                        best_k_cos = k_idx
                
                if best_k_cos != -1 and max_cos_score >= self.cell_cosine_strong_threshold:
                    target_col_info = current_defined_columns[best_k_cos]
                    is_horizontally_aligned_indicator = (target_col_info.get('xmin_col', -float('inf')) <= xc <= target_col_info.get('xmax_col', float('inf')))
                    is_vertically_in_row = (row_definition.get('ymin_row', -float('inf')) <= yc <= row_definition.get('ymax_row', float('inf')))

                    if is_horizontally_aligned_indicator and is_vertically_in_row:
                        # Solo añadir la palabra, las métricas se calcularán en TableAndFieldCoordinator
                        table_matrix_cells[r_idx][best_k_cos]['words'].append(word_dict_vec.copy())
                        table_matrix_cells[r_idx][best_k_cos]['assignment_confidence'] = "high_cosine"
                        processed_word_original_ids.add(word_id_key)
        
        # Iteración 2: Asignación de Respaldo (IoU)
        for r_idx, row_definition in enumerate(current_data_rows):
            for word_dict_vec in row_definition.get('words_in_row', []):
                word_id_key = word_dict_vec.get('original_id_for_table', word_dict_vec.get('internal_id', str(id(word_dict_vec))))
                if word_id_key in processed_word_original_ids:
                    continue

                # Asumir que la palabra es válida (ya filtrada por densidad si es necesario por el coordinador)
                word_poly_coords = self._get_word_polygon_for_iou(word_dict_vec)
                if not word_poly_coords:
                    continue
                
                best_k_overlap = -1
                max_iou_score = 0.0
                for k_idx, col_info in enumerate(current_defined_columns):
                    cell_xmin, cell_xmax = col_info.get('xmin_col'), col_info.get('xmax_col')
                    cell_ymin, cell_ymax = row_definition.get('ymin_row'), row_definition.get('ymax_row')
                    if None in [cell_xmin, cell_xmax, cell_ymin, cell_ymax] or cell_xmax <= cell_xmin or cell_ymax <= cell_ymin:
                        continue
                    cell_poly_coords = [[cell_xmin, cell_ymin], [cell_xmax, cell_ymin], [cell_xmax, cell_ymax], [cell_xmin, cell_ymax]]
                    current_iou = calculate_iou(word_poly_coords, cell_poly_coords)
                    if current_iou > max_iou_score:
                        max_iou_score = current_iou
                        best_k_overlap = k_idx
                
                if best_k_overlap != -1 and max_iou_score >= self.fallback_cell_iou_threshold:
                    table_matrix_cells[r_idx][best_k_overlap]['words'].append(word_dict_vec.copy())
                    if table_matrix_cells[r_idx][best_k_overlap]['assignment_confidence'] == "none":
                        table_matrix_cells[r_idx][best_k_overlap]['assignment_confidence'] = "medium_spatial_fallback"
                    processed_word_original_ids.add(word_id_key)
        
        # Ensamblar el texto de cada celda
        for r in range(num_rows_m):
            for k in range(num_cols_n):
                cell_content = table_matrix_cells[r][k]
                if cell_content['words']:
                    cell_content['words'].sort(
                        key = lambda w_vec: w_vec.get('original_geometry_data', {}).get('xmin',
                                        w_vec.get('vector_representation', [0,0])[0]
                                        if w_vec.get('vector_representation') and isinstance(w_vec['vector_representation'], list) and len(w_vec['vector_representation']) > 0 and isinstance(w_vec['vector_representation'][0], (int,float)) else 0
                                        )
                    )
                    cell_content['text'] = " ".join([w.get('text', '') for w in cell_content['words'] if 'text' in w])
        
        logger.debug(f"Asignación de celdas (simplificada) completada. {len(processed_word_original_ids)} palabras procesadas.")
        return table_matrix_cells
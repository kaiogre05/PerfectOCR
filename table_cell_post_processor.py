# PerfectOCR/core/table_extractor/table_cell_post_processor.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TableCellPostProcessor:
    def __init__(self, 
                spatial_validator_worker: Any, 
                cell_content_analyzer_worker: Any,
                post_processing_rules_config: Dict):
        self.rules_config = post_processing_rules_config
        self.spatial_validator = spatial_validator_worker
        self.cell_analyzer = cell_content_analyzer_worker
        
        # Cargar palabras clave desde la configuración para identificar tipos de columna
        # Se usarán versiones en mayúsculas para comparación insensible a mayúsculas/minúsculas
        self.value_column_keywords = {kw.upper() for kw in self.rules_config.get('value_column_keywords', [])}
        self.quantity_column_keywords = {kw.upper() for kw in self.rules_config.get('quantity_column_keywords', [])}
        self.currency_symbols = self.rules_config.get('currency_symbols', ['$', '€'])
        
        logger.debug(f"TableCellPostProcessor inicializado. Value Keywords: {self.value_column_keywords}, Quantity Keywords: {self.quantity_column_keywords}")

    def _is_primarily_numeric_or_currency(self, text: str) -> bool:
        """
        Determina si el texto es principalmente numérico o un símbolo de moneda.
        Maneja decimales con punto o coma, y miles con puntos o comas (siempre que no se mezclen de forma ambigua).
        """
        if not text or not isinstance(text, str):
            return False
        
        text_cleaned = text.strip()
        if not text_cleaned:
            return False

        # Comprobar si es solo un símbolo de moneda
        if text_cleaned in self.currency_symbols:
            return True

        # Quitar símbolos de moneda para el análisis numérico
        for symbol in self.currency_symbols:
            text_cleaned = text_cleaned.replace(symbol, '')
        text_cleaned = text_cleaned.strip()

        if not text_cleaned: # Si solo era un símbolo de moneda y ya se eliminó
            return True

        # Contar puntos y comas
        num_dots = text_cleaned.count('.')
        num_commas = text_cleaned.count(',')

        # Casos ambiguos (ej. 1.234,56 o 1,234.56 con múltiples separadores de miles y un decimal)
        # Si hay más de un punto Y más de una coma, es difícil determinar sin más contexto.
        # Por simplicidad, si ambos contadores son > 1, se considera no numérico.
        if num_dots > 1 and num_commas > 1:
            return False

        # Determinar el probable separador decimal y de miles
        # Asumimos que si hay múltiples comas, son separadores de miles y el punto es decimal.
        # Si hay múltiples puntos, son separadores de miles y la coma es decimal.
        # Si solo hay uno de cada, se asume que el último (más a la derecha) es el decimal.
        
        temp_text_for_check = text_cleaned
        
        if num_dots > 1: # Múltiples puntos, se asumen como separadores de miles. Coma es decimal.
            temp_text_for_check = temp_text_for_check.replace('.', '') # Quitar miles
            temp_text_for_check = temp_text_for_check.replace(',', '.', 1) # Convertir coma decimal a punto
        elif num_commas > 1: # Múltiples comas, se asumen como separadores de miles. Punto es decimal.
            temp_text_for_check = temp_text_for_check.replace(',', '') # Quitar miles
            # El punto ya es el separador decimal si existe
        elif num_dots == 1 and num_commas == 1: # Un punto y una coma
            # Si el punto está después de la coma, el punto es decimal. "1,234.56"
            # Si la coma está después del punto, la coma es decimal. "1.234,56" -> "1234.56"
            if temp_text_for_check.rfind('.') > temp_text_for_check.rfind(','):
                temp_text_for_check = temp_text_for_check.replace(',', '') # Quitar separador de miles
            else:
                temp_text_for_check = temp_text_for_check.replace('.', '') # Quitar separador de miles
                temp_text_for_check = temp_text_for_check.replace(',', '.', 1) # Convertir coma decimal a punto
        elif num_dots == 1: # Solo un punto, se asume decimal
            pass # El punto ya está bien
        elif num_commas == 1: # Solo una coma, se asume decimal
            temp_text_for_check = temp_text_for_check.replace(',', '.', 1) # Convertir a punto

        # Después de limpiar y estandarizar a punto decimal, verificar si es numérico
        try:
            float(temp_text_for_check)
            return True
        except ValueError:
            return False

    def process_cells(self,
                      assigned_table_matrix: List[List[Dict[str, Any]]],
                      defined_columns: List[Dict[str, Any]],
                      data_rows: List[Dict[str, Any]], # Se asume que las palabras aquí están devectorizadas
                      density_map: Optional[np.ndarray]) -> List[List[Dict[str, Any]]]:
        final_table_matrix_cells = []
        if not assigned_table_matrix:
            logger.warning("TableCellPostProcessor.process_cells: assigned_table_matrix está vacía.")
            return []
        if not defined_columns:
            logger.warning("TableCellPostProcessor.process_cells: defined_columns está vacía.")
            return assigned_table_matrix 

        num_cols = len(defined_columns)
        
        try: # Crear una copia para modificar
            modifiable_table_matrix = [[cell.copy() if isinstance(cell, dict) else {"text": str(cell) if cell is not None else "", "words": [], "assignment_confidence": "unknown", "cell_metrics": {}} for cell in row] for row in assigned_table_matrix]
        except Exception as e_copy:
            logger.error(f"Error creando copia de assigned_table_matrix: {e_copy}. Se procesará con riesgo.")
            modifiable_table_matrix = assigned_table_matrix


        for r_idx, row_of_cells in enumerate(modifiable_table_matrix):
            if r_idx >= len(data_rows):
                logger.warning(f"Índice de fila {r_idx} fuera de rango para data_rows ({len(data_rows)}).")
                final_table_matrix_cells.append(row_of_cells) 
                continue
            
            current_row_info = data_rows[r_idx]

            # Primera pasada: enriquecer métricas de celda (como estaba)
            for k_idx, cell_data in enumerate(row_of_cells):
                if not isinstance(cell_data, dict): # Asegurar que cell_data sea un dict
                    logger.warning(f"Celda ({r_idx},{k_idx}) no es un diccionario. Contenido: {cell_data}. Saltando enriquecimiento.")
                    row_of_cells[k_idx] = {"text": str(cell_data) if cell_data is not None else "", "words": [], "assignment_confidence": "unknown", "cell_metrics": {}}
                    cell_data = row_of_cells[k_idx]

                if 'cell_metrics' not in cell_data: cell_data['cell_metrics'] = {}
                
                current_cell_words = cell_data.get('words', [])
                if not isinstance(current_cell_words, list): current_cell_words = [] # Asegurar lista

                density_scores = [w.get('density_score', 0.0) for w in current_cell_words if isinstance(w, dict) and 'density_score' in w]
                if density_scores:
                    cell_data['cell_metrics']['avg_word_density_in_cell'] = round(float(np.mean(density_scores)), 3)
                
                if current_cell_words: # Solo analizar si hay palabras
                    structure_analysis = self.cell_analyzer.analyze_cell_structure(current_cell_words)
                    cell_data['cell_metrics'].update(structure_analysis)
                
                if not current_cell_words and density_map is not None:
                    if k_idx < num_cols:
                        col_info = defined_columns[k_idx]
                        xmin_col, ymin_row_val, xmax_col, ymax_row_val = col_info.get('xmin_col'), current_row_info.get('ymin_row'), col_info.get('xmax_col'), current_row_info.get('ymax_row')
                        if all(v is not None for v in [xmin_col, ymin_row_val, xmax_col, ymax_row_val]):
                            cell_bounds_geom = {'xmin': xmin_col, 'ymin': ymin_row_val, 'xmax': xmax_col, 'ymax': ymax_row_val}
                            cell_data['cell_metrics']['empty_cell_region_density'] = round(
                                self.spatial_validator.calculate_cell_region_density(cell_bounds_geom, density_map), 3)
            
            # Segunda pasada: Resegmentación estructural
            for k_idx_reseg, current_cell_for_reseg in enumerate(row_of_cells):
                if not isinstance(current_cell_for_reseg, dict): continue # Saltar si la celda no es un dict

                column_name_original = defined_columns[k_idx_reseg].get("H_k_text", "") if k_idx_reseg < num_cols else ""
                column_name_normalized = column_name_original.upper().strip()
                
                # Determinar tipo de columna usando las keywords cargadas
                is_value_col = any(kw in column_name_normalized for kw in self.value_column_keywords)
                is_quantity_col = any(kw in column_name_normalized for kw in self.quantity_column_keywords)

                words_in_current_cell = current_cell_for_reseg.get('words', [])
                if not isinstance(words_in_current_cell, list): words_in_current_cell = []
                
                words_in_current_cell.sort(key=lambda w: w.get('xmin', float('inf')) if isinstance(w, dict) else float('inf'))


                # REGLA 1: Columnas de VALOR (PRECIO, IMPORTE)
                if is_value_col and k_idx_reseg > 0: # Necesita una columna anterior para mover texto
                    words_to_keep_in_value_cell = []
                    words_to_move_to_prev_cell = []
                    
                    # Separar palabras numéricas/moneda de las no numéricas
                    numeric_found_in_cell = False
                    for i, word_obj in enumerate(words_in_current_cell):
                        word_text = word_obj.get('text', '')
                        if self._is_primarily_numeric_or_currency(word_text):
                            numeric_found_in_cell = True
                            # Si encontramos algo numérico, todo lo anterior que no lo era, se mueve
                            if words_to_keep_in_value_cell: # Hubo texto no numérico antes de este número
                                words_to_move_to_prev_cell.extend(words_to_keep_in_value_cell)
                                words_to_keep_in_value_cell = [] # Reset
                            words_to_keep_in_value_cell.append(word_obj)
                        else: # No es numérico
                            words_to_keep_in_value_cell.append(word_obj) # Temporalmente se queda, podría moverse si hay un numérico después

                    if not numeric_found_in_cell and words_in_current_cell: # Celda de valor sin nada numérico, mover todo
                        words_to_move_to_prev_cell = words_in_current_cell[:]
                        words_to_keep_in_value_cell = []
                        logger.debug(f"Reseg (Valor no numérico) F{r_idx}C{k_idx_reseg} '{column_name_original}': Moviendo TODO '{[w.get('text') for w in words_to_move_to_prev_cell]}' a celda anterior.")
                    elif words_to_move_to_prev_cell: # Solo si hay algo identificado explícitamente para mover
                        logger.debug(f"Reseg (Valor) F{r_idx}C{k_idx_reseg} '{column_name_original}': Moviendo '{[w.get('text') for w in words_to_move_to_prev_cell]}' a celda anterior. Queda: '{[w.get('text') for w in words_to_keep_in_value_cell]}'")


                    if words_to_move_to_prev_cell:
                        current_cell_for_reseg['words'] = words_to_keep_in_value_cell
                        current_cell_for_reseg['text'] = " ".join([w.get('text', '') for w in words_to_keep_in_value_cell]).strip()
                        
                        prev_cell_data = modifiable_table_matrix[r_idx][k_idx_reseg-1]
                        if not isinstance(prev_cell_data.get('words'), list): prev_cell_data['words'] = []
                        prev_cell_data['words'].extend(words_to_move_to_prev_cell)
                        prev_cell_data['words'].sort(key=lambda w: w.get('xmin', float('inf')) if isinstance(w,dict) else float('inf'))
                        prev_cell_data['text'] = " ".join([w.get('text', '') for w in prev_cell_data['words']]).strip()

                # REGLA 2: Columnas de CANTIDAD
                elif is_quantity_col and (k_idx_reseg + 1) < num_cols: # Necesita una columna siguiente para mover texto
                    words_to_keep_in_qty_cell = []
                    words_to_move_to_next_cell = []
                    numeric_part_ended_for_qty = False

                    for word_obj in words_in_current_cell:
                        word_text = word_obj.get('text', '')
                        # En columna de cantidad, solo queremos números. El resto se considera descripción.
                        if not numeric_part_ended_for_qty and word_text.isdigit(): # Solo dígitos puros para cantidad
                            words_to_keep_in_qty_cell.append(word_obj)
                        else:
                            numeric_part_ended_for_qty = True # Una vez que encontramos algo no-dígito, todo lo demás es descripción
                            words_to_move_to_next_cell.append(word_obj)
                    
                    if words_to_move_to_next_cell:
                        logger.debug(f"Reseg (Cantidad) F{r_idx}C{k_idx_reseg} '{column_name_original}': Moviendo '{[w.get('text') for w in words_to_move_to_next_cell]}' a celda siguiente (desc). Queda: '{[w.get('text') for w in words_to_keep_in_qty_cell]}'")
                        current_cell_for_reseg['words'] = words_to_keep_in_qty_cell
                        current_cell_for_reseg['text'] = " ".join([w.get('text', '') for w in words_to_keep_in_qty_cell]).strip()
                        
                        next_cell_data = modifiable_table_matrix[r_idx][k_idx_reseg+1]
                        if not isinstance(next_cell_data.get('words'), list): next_cell_data['words'] = []
                        
                        # Insertar al principio de la siguiente celda y re-sortear
                        current_next_cell_word_ids = {w.get('internal_id', id(w)) for w in next_cell_data['words'] if isinstance(w, dict)}
                        prepend_words = []
                        for w_move in words_to_move_to_next_cell:
                            if w_move.get('internal_id', id(w_move)) not in current_next_cell_word_ids:
                                prepend_words.append(w_move)
                        
                        next_cell_data['words'] = prepend_words + next_cell_data['words']
                        next_cell_data['words'].sort(key=lambda w: w.get('xmin', float('inf')) if isinstance(w,dict) else float('inf'))
                        next_cell_data['text'] = " ".join([w.get('text', '') for w in next_cell_data['words']]).strip()
            
            final_table_matrix_cells.append(row_of_cells)

        logger.info(f"Post-procesamiento de celdas (con resegmentación estructural) completado. Filas procesadas: {len(final_table_matrix_cells)}")
        return final_table_matrix_cells
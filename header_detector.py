# PerfectOCR/core/table_extractor/header_detector.py
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from utils.geometric import get_polygon_y_center, get_polygon_bounds
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

class HeaderDetector:
    def __init__(self,
                 config: Dict,
                 header_keywords_list: List[str], # Parámetro con la lista de keywords
                 page_dimensions: Optional[Dict[str, Any]] = None):
        
        self.config = config
        self.page_dimensions = page_dimensions or {} 
        if not self.page_dimensions: 
            logger.debug(f"HeaderDetector, Inicializado SIN page_dimensions efectivas. Esperando llamada explícita a set_page_dimensions.")

        self.header_keywords_list = [str(kw).upper().strip() for kw in header_keywords_list if kw]        
        self.header_fuzzy_min_ratio = float(self.config.get('header_detection_fuzzy_min_ratio', 85.0)) 
        self.min_y_ratio = float(self.config.get('header_min_y_ratio', 0.05))
        self.max_y_ratio = float(self.config.get('header_max_y_ratio', 0.75)) 
        self.min_keywords_in_line = int(self.config.get('min_header_keywords_in_line', 2)) 
        self.max_keywords_in_line = int(self.config.get('max_header_keywords_in_line', 5)) 
        self.min_line_confidence = float(self.config.get('min_line_confidence_for_header', 70.0)) 
        self.max_header_line_gap_factor = float(self.config.get('max_header_line_gap_factor', 2.5)) 
        self.default_line_height_for_gap = float(self.config.get('default_line_height_for_gap', 20.0)) 
        if not self.header_keywords_list:
            logger.error("HeaderDetector inicializado sin palabras clave de encabezado efectivas. La detección de encabezados fallará.")
        else:
            logger.info(f"HeaderDetector inicializado. Keywords cargadas: {len(self.header_keywords_list)}. Ejemplo: {self.header_keywords_list[:5]}")

    def set_page_dimensions(self, page_dimensions: Dict[str, Any]): #
        if page_dimensions and page_dimensions.get('width') and page_dimensions.get('height'): #
            self.page_dimensions = page_dimensions #
            self.page_height = int(self.page_dimensions['height']) #
            self.page_width = int(self.page_dimensions['width']) #
            logger.debug(f"HeaderDetector: Dimensiones de página establecidas a {self.page_dimensions}")
        else:
            logger.warning(f"HeaderDetector: Intento de establecer dimensiones de página inválidas: {page_dimensions}")

    def _get_words_from_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]: #
        all_words = [] #
        for line in lines:  #
            words_in_line = line.get('constituent_elements', []) #
            for word in words_in_line: #
                all_words.append(word) #
        return all_words

    def _is_line_a_header_candidate(self, line_obj: Dict[str, Any]) -> bool: #
        line_text_preview_for_debug = str(line_obj.get('text_raw', 'N/A'))[:70] #

        if not isinstance(line_obj, dict): #
            logger.error(f"HeaderDetector._is_line_a_header_candidate: 'line_obj' no es un diccionario. Recibido: {type(line_obj)}")
            return False

        current_page_height = self.page_dimensions.get('height') #
        if not isinstance(current_page_height, (int, float)) or current_page_height <= 0: #
            logger.warning(
                f"'{line_text_preview_for_debug}': page_dimensions.height no es válida ({current_page_height}). "
                f"Current page_dimensions: {self.page_dimensions}. No se puede validar zona Y."
            )
            return False 

        line_polygon = line_obj.get('polygon_final') #
        if not line_polygon: #
            return False
            
        try:
            line_y_center = get_polygon_y_center(line_polygon) #
        except Exception as e_ycenter:
            logger.warning(f"'{line_text_preview_for_debug}': Excepción en get_polygon_y_center: {e_ycenter}. Coords: {line_polygon}")
            return False
        
        min_y_allowed = current_page_height * self.min_y_ratio #
        max_y_allowed = current_page_height * self.max_y_ratio #

        if not (min_y_allowed <= line_y_center <= max_y_allowed): #
            return False

        keyword_count = 0 #
        constituent_elements = line_obj.get('constituent_elements', []) #
        if not constituent_elements:  #
            return False

        if not self.header_keywords_list:  #
            logger.warning(f"'{line_text_preview_for_debug}': Lista de header_keywords_list está vacía. No se pueden encontrar keywords.")
            return False


        for idx, elem_dict in enumerate(constituent_elements): #
            text_upper = elem_dict.get('text', ' ').upper().strip()  #
            if not text_upper:  #
                continue
            
            is_exact_match = text_upper in self.header_keywords_list #
            
            if is_exact_match: #
                keyword_count += 1 #
            else:
                match_result_fuzzy = process.extractOne(text_upper, self.header_keywords_list, scorer=fuzz.WRatio, score_cutoff=self.header_fuzzy_min_ratio) #
                if match_result_fuzzy: #
                    keyword_count += 1 #
        
        passes_min_keywords = keyword_count >= self.min_keywords_in_line #
        passes_max_keywords = keyword_count <= self.max_keywords_in_line  #
        current_line_avg_conf = line_obj.get('structural_confidence_pct', 0.0)  #
        passes_confidence = current_line_avg_conf >= self.min_line_confidence #
        
        final_decision = passes_min_keywords and passes_max_keywords and passes_confidence #
        return final_decision

    def identify_header_band_and_words(self, formed_lines: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[float], Optional[float]]: #
        current_page_height = self.page_dimensions.get('height') #
        current_page_width = self.page_dimensions.get('width') #
        if not self.page_dimensions or \
           not isinstance(current_page_height, (int, float)) or current_page_height <= 0 or \
           not isinstance(current_page_width, (int, float)) or current_page_width <= 0: #
            logger.error(
                f"HeaderDetector.identify_header_band_and_words: No se pueden identificar encabezados sin "
                f"dimensiones de página válidas (alto: {current_page_height}, ancho: {current_page_width}). "
                f"page_dimensions dict: {self.page_dimensions}. Abortando detección de banda."
            )
            
            return None, None, None
        if not self.header_keywords_list:
            logger.error("HeaderDetector.identify_header_band_and_words: La lista de palabras clave de encabezado está vacía. No se puede proceder.")
            return None, None, None
        
        potential_header_lines = [] #
        for line_idx, line_to_check in enumerate(formed_lines): 
            if self._is_line_a_header_candidate(line_to_check):  
                potential_header_lines.append(line_to_check) 
        
        if not potential_header_lines: #
            logger.warning("No se encontraron líneas de encabezado potenciales después de filtrar con _is_line_a_header_candidate.") 
            return None, None, None

        potential_header_lines.sort(key=lambda l: get_polygon_y_center(l.get('polygon_final', []))) #
        
        final_header_lines_block: List[Dict[str,Any]] = [] #
        if potential_header_lines: #
            final_header_lines_block.append(potential_header_lines[0]) #
            if len(potential_header_lines) > 1: #
                last_line_in_block = potential_header_lines[0] #
                avg_h = self.default_line_height_for_gap #
                
                candidate_heights = [] #
                for cand_line in potential_header_lines: #
                    poly = cand_line.get('polygon_final') #
                    if poly: #
                        try:
                            _, ymin_cand, _, ymax_cand = get_polygon_bounds(poly) #
                            if ymax_cand > ymin_cand: #
                                candidate_heights.append(ymax_cand - ymin_cand) #
                        except:
                            pass
                if candidate_heights: #
                    avg_h = np.median(candidate_heights) if candidate_heights else self.default_line_height_for_gap #
                    if avg_h <=0: avg_h = self.default_line_height_for_gap #
                logger.debug(f"Agrupando líneas de encabezado. Altura de línea mediana calculada para gap: {avg_h:.2f}")


                for next_line_idx, next_line in enumerate(potential_header_lines[1:]): #
                    next_line_poly_for_gap = next_line.get('polygon_final') #
                    last_line_poly_for_gap = last_line_in_block.get('polygon_final')  #
                    
                    if not next_line_poly_for_gap or not last_line_poly_for_gap:  #
                        continue
                    try:
                        next_y_min = get_polygon_bounds(next_line_poly_for_gap)[1] #
                        last_y_max = get_polygon_bounds(last_line_poly_for_gap)[3] #
                        gap = next_y_min - last_y_max #
                        if gap < (avg_h * self.max_header_line_gap_factor): #
                            final_header_lines_block.append(next_line) #
                            last_line_in_block = next_line  #
                        else: 
                            break 
                    except Exception as e_gap: 
                        logger.warning(f"Excepción calculando gap de línea de encabezado: {e_gap}."); 
                        break 

        if not final_header_lines_block: #
            logger.warning("El bloque final de líneas de encabezado está vacío después de la agrupación."); return None, None, None
        
        header_words = self._get_words_from_lines(final_header_lines_block) #
        if not header_words: #
            logger.warning("No se extrajeron palabras del bloque de encabezados."); return None, None, None
        
        all_ymins, all_ymaxs = [], [] #
        for line in final_header_lines_block: #
            line_poly_for_bounds = line.get('polygon_final') #
            if line_poly_for_bounds: #
                try:
                    _, ymin_b, _, ymax_b = get_polygon_bounds(line_poly_for_bounds) #
                    all_ymins.append(ymin_b); all_ymaxs.append(ymax_b) #
                except Exception as e_bounds: logger.warning(f"No se pudieron obtener bounds para la línea: {e_bounds}")
        
        y_min_band = min(all_ymins) if all_ymins else None #
        y_max_band = max(all_ymaxs) if all_ymaxs else None #
        
        if y_min_band is None or y_max_band is None: #
            logger.warning("No se pudieron determinar los límites Y de la banda de encabezados.")
            return header_words, None, None 
            
        logger.info(f"Banda de encabezados identificada entre Y={y_min_band:.2f} y Y={y_max_band:.2f} con {len(header_words)} palabras.")
        return header_words, y_min_band, y_max_band
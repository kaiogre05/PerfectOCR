# PerfectOCR/core/lineal_finder/lineal_reconstructor_worker.py
import logging
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from shapely.geometry import Polygon
from shapely.ops import unary_union
from utils.geometric import get_shapely_polygon, get_polygon_bounds #

logger = logging.getLogger(__name__)

class LineReconstructorWorker:
    def __init__(self, page_width: int, page_height: int, config: Dict):
        self.config = config
        self.page_width = page_width
        self.page_height = page_height

        assignment_cfg = self.config.get('paddle_tesseract_assignment_params', {}) 
        self.min_vertical_overlap_ratio = float(assignment_cfg.get('min_vertical_overlap_ratio', {})) 
        self.min_iou_for_assignment = float(assignment_cfg.get('min_iou_for_assignment', {})) 
        self.assign_if_tess_centroid_in_paddle_line = bool(assignment_cfg.get('assign_if_tess_centroid_in_paddle_line', {})) 
        self.paddle_line_x_expansion_factor_for_centroid = float(assignment_cfg.get('paddle_line_x_expansion_factor_for_centroid', {})) 

        line_recon_cfg = self.config.get('line_reconstructor_params', {})
        self.dynamic_origin_h_threshold = float(line_recon_cfg.get('dynamic_origin_height_threshold', 2800.00))
        self.y_centroid_alignment_ratio_threshold = float(line_recon_cfg.get('y_centroid_alignment_ratio_threshold', {}))
        self.y_centroid_abs_diff_threshold_px = float(line_recon_cfg.get('y_centroid_abs_diff_threshold_px', {}))

        self.element_fusion_cfg = self.config.get('element_fusion_params', {})
        self.element_iou_threshold = float(self.element_fusion_cfg.get('iou_threshold_for_fusion', {}))
        self.element_text_sim_threshold = float(self.element_fusion_cfg.get('text_similarity_for_strong_match', {}))
        self.paddle_confidence_normalization_factor = float(self.element_fusion_cfg.get('paddle_confidence_normalization_factor', {}))
        self.min_tess_conf_for_text_priority = float(self.element_fusion_cfg.get('min_tess_confidence_for_text_priority', {}))
        self.paddle_high_conf_threshold = float(self.element_fusion_cfg.get('paddle_high_confidence_threshold', {}))

        self.focus_ox, self.focus_oy = self._determine_focus_point()
        
        logger.info(
            f"LineReconstructorWorker inicializado. Page: {page_width}x{page_height}. "
            f"Assign Params: V.Overlap={self.min_vertical_overlap_ratio}, IoU={self.min_iou_for_assignment}, "
            f"CentroidInPaddle={self.assign_if_tess_centroid_in_paddle_line}, PaddleXExpand={self.paddle_line_x_expansion_factor_for_centroid}. "
            f"LineRecon Params (Orphan): DynOriginHThresh={self.dynamic_origin_h_threshold}, YAlignRatio={self.y_centroid_alignment_ratio_threshold}, YAlignAbs={self.y_centroid_abs_diff_threshold_px}."
            f"Fusion Logic Params: MinTessConfForPriority={self.min_tess_conf_for_text_priority}, PaddleHighConfThresh={self.paddle_high_conf_threshold}."
        ) #

    def _determine_focus_point(self) -> Tuple[float, float]:
        if self.page_height <= self.dynamic_origin_h_threshold: 
            origin_x, origin_y = 0.0, float(self.page_height) 
        else:
            origin_x, origin_y = float(self.page_width) / 2.0, float(self.page_height) / 2.0 
        return origin_x, origin_y

    def _normalize_raw_confidence(self, raw_confidence: Union[float, int, str], engine_name: str) -> float:
        try:
            norm_conf = float(raw_confidence)
        except (ValueError, TypeError):
            logger.warning(f"Confianza cruda inválida '{raw_confidence}' para {engine_name}, usando 0.0.")
            return 0.0

        return max(0.0, min(100.0, norm_conf))

    def _prepare_element_for_fusion(self, ocr_item: Dict, element_idx: int, engine_name: str, item_type: str) -> Optional[Dict]:
        text = str(ocr_item.get('text', '')).strip() #
        if not text: 
            return None

        raw_confidence = ocr_item.get('confidence', 0.0)
        normalized_confidence = self._normalize_raw_confidence(raw_confidence, engine_name)

        poly_coords_raw: Optional[List[List[Union[int, float]]]] = None
        
        if engine_name == 'tesseract' and item_type == 'word' and 'bbox' in ocr_item:
            bbox = ocr_item['bbox'] #
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    if x2 <= x1 or y2 <= y1:
                        logger.debug(f"Tesseract bbox inválido para ID {element_idx}, texto '{text}': {bbox}")
                        return None
                    poly_coords_raw = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                except (TypeError, ValueError):
                    logger.warning(f"Tesseract bbox con coordenadas no numéricas para ID {element_idx}, texto '{text}': {bbox}")
                    return None
            else:
                logger.debug(f"Tesseract bbox con formato incorrecto para ID {element_idx}, texto '{text}': {bbox}")
                return None
        elif engine_name == 'paddleocr' and item_type == 'line' and 'polygon_coords' in ocr_item:
            poly_coords_raw = ocr_item['polygon_coords']
        
        if not poly_coords_raw:
            logger.debug(f"Elemento {engine_name} ID {element_idx} sin geometría válida: '{text}'")
            return None

        try:
            float_poly_coords = [[float(p[0]), float(p[1])] for p_idx, p in enumerate(poly_coords_raw)
                                if isinstance(p, (list, tuple)) and len(p) == 2]
            
            if len(float_poly_coords) < 3:
                logger.debug(f"Coordenadas insuficientes o inválidas para polígono {engine_name} ID {element_idx}, texto '{text}': {float_poly_coords}")
                return None

            shapely_poly = get_shapely_polygon(float_poly_coords)
            if not shapely_poly: 
                logger.debug(f"Polígono Shapely inválido para {engine_name} ID {element_idx}, texto '{text}' desde coords: {float_poly_coords}")
                return None

            min_x, min_y, max_x, max_y = shapely_poly.bounds
            height = max_y - min_y
            width = max_x - min_x
            
            if height <= 0.01 or width <= 0.01:
                logger.debug(f"Elemento {engine_name} ID {element_idx} con altura/anchura no positiva: H={height:.2f}, W={width:.2f}")
                return None
            
            return {
                "internal_id": f"{engine_name}_{item_type}_{element_idx:04d}",
                "text_raw": text,
                "shapely_polygon": shapely_poly,
                "polygon_coords": float_poly_coords,
                "confidence": normalized_confidence,
                "engine_source": engine_name,
                "original_ocr_data": ocr_item.copy(),
                "cx": float(shapely_poly.centroid.x), "cy": float(shapely_poly.centroid.y),
                "xmin": float(min_x), "ymin": float(min_y), "xmax": float(max_x), "ymax": float(max_y),
                "height": float(height), "width": float(width)
            }
        except Exception as e:
            logger.error(f"Excepción preparando elemento para fusión {engine_name} ID {element_idx}, texto '{text}': {e}", exc_info=False)
            return None

    def _assign_tesseract_words_to_paddle_line(self, paddle_line_segment: Dict, tesseract_words: List[Dict]) -> List[Dict]:
        assigned_words = []
        p_line_shapely_poly = paddle_line_segment.get("shapely_polygon")
        if not p_line_shapely_poly: return assigned_words 

        p_xmin, p_ymin, p_xmax, p_ymax = p_line_shapely_poly.bounds
        p_width = p_xmax - p_xmin
        expanded_xmin = p_xmin - (p_width * (self.paddle_line_x_expansion_factor_for_centroid - 1.0) / 2.0)
        expanded_xmax = p_xmax + (p_width * (self.paddle_line_x_expansion_factor_for_centroid - 1.0) / 2.0)
        search_poly_coords = [[expanded_xmin, p_ymin], [expanded_xmax, p_ymin], [expanded_xmax, p_ymax], [expanded_xmin, p_ymax]]
        p_line_search_poly = get_shapely_polygon(search_poly_coords)
        if not p_line_search_poly : p_line_search_poly = p_line_shapely_poly

        for t_word in tesseract_words:
            t_word_shapely_poly = t_word.get("shapely_polygon")
            if not t_word_shapely_poly: continue

            assigned_by_iou = False #
            assigned_by_centroid = False #

            try:
                iou = p_line_shapely_poly.intersection(t_word_shapely_poly).area / t_word_shapely_poly.area if t_word_shapely_poly.area > 0 else 0 #
                if iou >= self.min_iou_for_assignment: #
                    assigned_by_iou = True #
                
                if self.assign_if_tess_centroid_in_paddle_line: #
                    t_word_centroid = t_word_shapely_poly.centroid #
                    if p_line_search_poly.contains(t_word_centroid): #
                       assigned_by_centroid = True #
                
                overlap_y = max(0, min(t_word.get("ymax",0), paddle_line_segment.get("ymax",0)) - \
                                   max(t_word.get("ymin",0), paddle_line_segment.get("ymin",0))) #
                t_word_height = t_word.get("height", 1.0) #
                vertical_overlap_ratio_word = overlap_y / t_word_height if t_word_height > 0 else 0 #
                
                if assigned_by_iou and (vertical_overlap_ratio_word >= self.min_vertical_overlap_ratio): #
                    assigned_words.append(t_word) #
                elif not assigned_by_iou and assigned_by_centroid and (vertical_overlap_ratio_word >= self.min_vertical_overlap_ratio): #
                    assigned_words.append(t_word) #
                    
            except Exception as e:
                logger.warning(f"Error calculando asignación para palabra {t_word.get('internal_id','')}: {e}", exc_info=False)
                continue
    
        assigned_words.sort(key=lambda el: el.get("xmin", 0.0)) #
        return assigned_words

    def _build_line_output_from_generic_constituents(self, line_constituents: List[Dict], line_id: str, fusion_source_info: str) -> Optional[Dict[str, Any]]:
        if not line_constituents: return None
        
        line_constituents.sort(key=lambda el: el.get("xmin", 0.0)) #
        line_text_raw = " ".join([el.get("text_raw", "") for el in line_constituents]).strip() #
        if not line_text_raw: return None

        all_constituent_polys_shapely = [el.get("shapely_polygon") for el in line_constituents if el.get("shapely_polygon")] #
        line_bbox_coords_list: List[List[float]] = [] #
        geom_props = {} #

        if all_constituent_polys_shapely:
            valid_polys = [p for p in all_constituent_polys_shapely if p and not p.is_empty and p.is_valid] #
            if valid_polys:
                try:
                    unioned_geom = unary_union(valid_polys) #
                    if unioned_geom and not unioned_geom.is_empty: #
                        line_polygon_for_bounds = unioned_geom.convex_hull if unioned_geom.geom_type != 'Point' else unioned_geom.buffer(0.1).convex_hull #
                        if not line_polygon_for_bounds.is_valid or line_polygon_for_bounds.is_empty: #
                            line_polygon_for_bounds = unioned_geom.envelope #
                        
                        min_x_u, min_y_u, max_x_u, max_y_u = line_polygon_for_bounds.bounds #
                        line_bbox_coords_list = [[min_x_u, min_y_u], [max_x_u, min_y_u], [max_x_u, max_y_u], [min_x_u, max_y_u]] #
                        
                        geom_props = {
                            "cy_avg": float(line_polygon_for_bounds.centroid.y), #
                            "ymin_line": float(min_y_u), "ymax_line": float(max_y_u), "height_line": float(max_y_u - min_y_u), #
                            "xmin_line": float(min_x_u), "xmax_line": float(max_x_u), "width_line": float(max_x_u - min_x_u) #
                        }
                except Exception as e_union:
                    logger.warning(f"Error uniendo polígonos para línea {line_id} ({fusion_source_info}): {e_union}. Usando MBR de constituyentes.")
        
        if not line_bbox_coords_list and line_constituents: #
            all_xmins = [el.get("xmin") for el in line_constituents if el.get("xmin") is not None] #
            all_ymins = [el.get("ymin") for el in line_constituents if el.get("ymin") is not None] #
            all_xmaxs = [el.get("xmax") for el in line_constituents if el.get("xmax") is not None] #
            all_ymaxs = [el.get("ymax") for el in line_constituents if el.get("ymax") is not None] #
            if all_xmins and all_ymins and all_xmaxs and all_ymaxs: #
                min_x,max_x = min(all_xmins), max(all_xmaxs) #
                min_y,max_y = min(all_ymins), max(all_ymaxs) #
                line_bbox_coords_list = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]] #
                geom_props = {
                    "cy_avg": (min_y + max_y) / 2.0, #
                    "ymin_line": min_y, "ymax_line": max_y, "height_line": max_y - min_y, #
                    "xmin_line": min_x, "xmax_line": max_x, "width_line": max_x - min_x #
                }

        confidences = [el.get("confidence", 0.0) for el in line_constituents if el.get("confidence") is not None] #
        avg_confidence = np.mean(confidences) if confidences else 0.0 #
        
        output_constituent_elements = [] #
        for el_dict in line_constituents:
             output_constituent_elements.append({
                "text": el_dict.get("text_raw",""), #
                "polygon_coords": el_dict.get("polygon_coords", []), #
                "confidence": el_dict.get("confidence",0.0), #
                "source_info": el_dict.get("engine_source", "unknown_constituent"), #
                "internal_id": el_dict.get("internal_id"), #
                "cx": el_dict.get("cx"), "cy": el_dict.get("cy"), #
                "xmin": el_dict.get("xmin"), "ymin": el_dict.get("ymin"), #
                "xmax": el_dict.get("xmax"), "ymax": el_dict.get("ymax"), #
                "height": el_dict.get("height"), "width": el_dict.get("width"), #
                "original_id_for_table": el_dict.get("internal_id") #
            })

        return {
            "line_id": line_id, #
            "text_raw": line_text_raw, #
            "polygon_line_bbox": line_bbox_coords_list,  #
            "avg_constituent_confidence": round(float(avg_confidence), 2), #
            "constituent_elements_ocr_data": output_constituent_elements, #
            "geometric_properties_line": geom_props, #
            "fusion_source": fusion_source_info #
        }

    def _build_line_from_paddle_and_tesseract_constituents(self,
                                                          paddle_line_segment: Dict,
                                                          tesseract_constituents: List[Dict],
                                                          line_id_prefix: str,
                                                          line_idx: int) -> Dict[str, Any]:
        line_text_raw = " " #
        avg_confidence = 0.0 #
        fusion_source_info = "unknown_fusion_outcome" #

        paddle_line_text_original = paddle_line_segment.get("text_raw", " ") #
        paddle_line_confidence_original = paddle_line_segment.get("confidence", 0.0) #

        if tesseract_constituents: #
            tess_texts_list = [el.get("text_raw", "") for el in tesseract_constituents] #
            line_text_from_tess = " ".join(tess_texts_list).strip() #

            tess_confidences_list = [el.get("confidence", 0.0) for el in tesseract_constituents if el.get("confidence") is not None] #
            avg_tess_confidence = np.mean(tess_confidences_list) if tess_confidences_list else 0.0 #

            # Renamed for clarity
            use_paddle_text_and_conf_decision = False #
            if avg_tess_confidence < self.min_tess_conf_for_text_priority and \
               paddle_line_confidence_original >= self.paddle_high_conf_threshold: #
                
                if line_text_from_tess and not paddle_line_text_original.strip(): #
                    logger.debug(f"Linea {line_id_prefix}_{line_idx:04d}: Tesseract tiene texto ('{line_text_from_tess[:50]}...') y confianza baja ({avg_tess_confidence:.2f}), "
                                f"Paddle tiene confianza alta ({paddle_line_confidence_original:.2f}) pero texto vacío. Manteniendo texto de Tesseract.")
                    use_paddle_text_and_conf_decision = False #
                else:
                    use_paddle_text_and_conf_decision = True #
                    logger.info(f"Linea {line_id_prefix}_{line_idx:04d}: Prefiriendo texto/confianza de Paddle ({paddle_line_confidence_original:.2f}%, '{paddle_line_text_original[:50]}...') sobre Tesseract ({avg_tess_confidence:.2f}%, '{line_text_from_tess[:50]}...').")

            if use_paddle_text_and_conf_decision: #
                line_text_raw = paddle_line_text_original #
                avg_confidence = paddle_line_confidence_original #
                fusion_source_info = "paddle_text_preferred_low_tess_conf" #
            else:
                line_text_raw = line_text_from_tess #
                avg_confidence = avg_tess_confidence #
                fusion_source_info = "paddle_geom_tess_words"  #
        else:
            line_text_raw = paddle_line_text_original #
            avg_confidence = paddle_line_confidence_original #
            fusion_source_info = "paddle_geom_only" #
        
        if not line_text_raw.strip(): #
            current_tess_text_combined = " ".join([el.get("text_raw", "") for el in tesseract_constituents]).strip() if tesseract_constituents else "" #
            if current_tess_text_combined: #
                line_text_raw = current_tess_text_combined #
                confidences_fallback_tess = [el.get("confidence", 0.0) for el in tesseract_constituents if el.get("confidence") is not None] #
                avg_confidence = np.mean(confidences_fallback_tess) if confidences_fallback_tess else 0.0 #
                fusion_source_info += "_fb_tess" #
                logger.debug(f"Linea {line_id_prefix}_{line_idx:04d}: Usando texto de Tesseract como fallback (selección principal vacía).")
            elif paddle_line_text_original.strip(): #
                line_text_raw = paddle_line_text_original #
                avg_confidence = paddle_line_confidence_original #
                fusion_source_info += "_fb_paddle" #
                logger.debug(f"Linea {line_id_prefix}_{line_idx:04d}: Usando texto de Paddle como fallback (selección principal vacía).")

        output_constituent_elements = [] #
        if tesseract_constituents: #
            for t_elem_dict in tesseract_constituents:
                output_constituent_elements.append({
                    "text": t_elem_dict.get("text_raw",""), #
                    "polygon_coords": t_elem_dict.get("polygon_coords", []), #
                    "confidence": t_elem_dict.get("confidence",0.0), #
                    "source_info": "tesseract_in_paddle_line", #
                    "internal_id": t_elem_dict.get("internal_id"), #
                    "cx": t_elem_dict.get("cx"), "cy": t_elem_dict.get("cy"), #
                    "xmin": t_elem_dict.get("xmin"), "ymin": t_elem_dict.get("ymin"), #
                    "xmax": t_elem_dict.get("xmax"), "ymax": t_elem_dict.get("ymax"), #
                    "height": t_elem_dict.get("height"), "width": t_elem_dict.get("width"), #
                    "original_id_for_table": t_elem_dict.get("internal_id") #
                })
        elif paddle_line_segment:  #
             output_constituent_elements.append({
                "text": paddle_line_segment.get("text_raw",""), #
                "polygon_coords": paddle_line_segment.get("polygon_coords", []), #
                "confidence": paddle_line_segment.get("confidence",0.0), #
                "source_info": "paddle_line_only", #
                "internal_id": paddle_line_segment.get("internal_id"), #
                "cx": paddle_line_segment.get("cx"), "cy": paddle_line_segment.get("cy"), #
                "xmin": paddle_line_segment.get("xmin"), "ymin": paddle_line_segment.get("ymin"), #
                "xmax": paddle_line_segment.get("xmax"), "ymax": paddle_line_segment.get("ymax"), #
                "height": paddle_line_segment.get("height"), "width": paddle_line_segment.get("width"), #
                "original_id_for_table": paddle_line_segment.get("internal_id") #
            })
             
        line_geom_props = {
            "cy_avg": paddle_line_segment.get("cy"), #
            "ymin_line": paddle_line_segment.get("ymin"), "ymax_line": paddle_line_segment.get("ymax"), #
            "height_line": paddle_line_segment.get("height"), #
            "xmin_line": paddle_line_segment.get("xmin"), "xmax_line": paddle_line_segment.get("xmax"), #
            "width_line": paddle_line_segment.get("width") #
        }

        return {
            "line_id": f"{line_id_prefix}_{line_idx:04d}", #
            "text_raw": line_text_raw, #
            "polygon_line_bbox": paddle_line_segment.get("polygon_coords", []), #
            "avg_constituent_confidence": round(float(avg_confidence), 2), #
            "constituent_elements_ocr_data": output_constituent_elements, #
            "geometric_properties_line": line_geom_props, #
            "fusion_source": fusion_source_info #
        }

    def _reconstruct_orphan_tesseract_lines(self, orphan_tess_elements: List[Dict], start_line_idx: int) -> List[Dict]:
        formed_lines = [] #
        if not orphan_tess_elements: #
            return formed_lines

        orphan_tess_elements.sort(key=lambda el: (el.get("ymin", 0.0), el.get("xmin", 0.0))) #
        
        current_line_constituents: List[Dict] = [] #
        if orphan_tess_elements: #
            current_line_constituents.append(orphan_tess_elements[0]) #
            for i in range(1, len(orphan_tess_elements)):
                current_elem = orphan_tess_elements[i] #
                ref_elem_in_line = current_line_constituents[0] #
                
                ref_elem_height = ref_elem_in_line.get("height", self.y_centroid_abs_diff_threshold_px) #
                if ref_elem_height <= 0.1: ref_elem_height = self.y_centroid_abs_diff_threshold_px #
                
                threshold_y_distance_ratio = self.y_centroid_alignment_ratio_threshold * ref_elem_height #
                threshold_y_distance_abs = self.y_centroid_abs_diff_threshold_px / 2.0 #
                effective_y_threshold = max(threshold_y_distance_ratio, threshold_y_distance_abs) #
                if abs(current_elem.get("cy", 0.0) - ref_elem_in_line.get("cy", 0.0)) < effective_y_threshold: #
                    current_line_constituents.append(current_elem) #
                else: 
                    if current_line_constituents: #
                        line_obj = self._build_line_output_from_generic_constituents(
                            current_line_constituents, 
                            f"orphan_tess_line_{start_line_idx + len(formed_lines):04d}", 
                            "tesseract_orphan_reconstruction"
                        ) #
                        if line_obj: formed_lines.append(line_obj) #
                    current_line_constituents = [current_elem] #
        
            if current_line_constituents:  #
                line_obj = self._build_line_output_from_generic_constituents(
                    current_line_constituents, 
                    f"orphan_tess_line_{start_line_idx + len(formed_lines):04d}", 
                    "tesseract_orphan_reconstruction"
                ) #
                if line_obj: formed_lines.append(line_obj) #
        
        return formed_lines


    def fuse_and_reconstruct_all_lines(self, tesseract_raw_words: List[Dict], paddle_raw_segments: List[Dict]) -> List[Dict[str, Any]]:
        logger.info("Iniciando reconstrucción de líneas (Paddle guía, Tesseract conforma).")

        prepared_tess_elements = [] #
        for i, item in enumerate(tesseract_raw_words):
            elem = self._prepare_element_for_fusion(item, i, "tesseract", "word") #
            if elem: prepared_tess_elements.append(elem) #
        logger.info(f"Elementos Tesseract preparados y validados: {len(prepared_tess_elements)}")

        prepared_padd_line_segments = [] #
        for i, item in enumerate(paddle_raw_segments):
            elem = self._prepare_element_for_fusion(item, i, "paddleocr", "line") #
            if elem: prepared_padd_line_segments.append(elem) #
        logger.info(f"Segmentos de línea PaddleOCR preparados y validados: {len(prepared_padd_line_segments)}")

        final_reconstructed_lines: List[Dict[str, Any]] = [] #
        used_tesseract_word_ids = set() #

        if prepared_padd_line_segments: #
            prepared_padd_line_segments.sort(key=lambda pl: (pl.get("ymin", 0.0), pl.get("xmin", 0.0))) #
            
            for paddle_line_idx, p_line_segment in enumerate(prepared_padd_line_segments):
                available_tess_words = [tw for tw in prepared_tess_elements if tw.get("internal_id") not in used_tesseract_word_ids] #
                
                constituent_tess_words = self._assign_tesseract_words_to_paddle_line(p_line_segment, available_tess_words) #
                
                for t_word in constituent_tess_words: #
                    used_tesseract_word_ids.add(t_word.get("internal_id")) #

                line_obj = self._build_line_from_paddle_and_tesseract_constituents(
                    p_line_segment, 
                    constituent_tess_words, 
                    "ppl_fused", 
                    paddle_line_idx
                ) #
                if line_obj and line_obj.get("text_raw","").strip(): #
                    final_reconstructed_lines.append(line_obj) #
        else:
            logger.info("No hay segmentos de PaddleOCR para guiar la reconstrucción. Se procederá solo con Tesseract si hay palabras.")

        orphan_tess_elements = [t_word for t_word in prepared_tess_elements if t_word.get("internal_id") not in used_tesseract_word_ids] #
        if orphan_tess_elements: #
            logger.info(f"Procesando {len(orphan_tess_elements)} palabras Tesseract huérfanas.")
            orphan_lines = self._reconstruct_orphan_tesseract_lines(orphan_tess_elements, len(final_reconstructed_lines)) #
            final_reconstructed_lines.extend(orphan_lines) #
        
        final_reconstructed_lines.sort(key=lambda l: (l.get("geometric_properties_line", {}).get("ymin_line", float('inf')), 
                                                      l.get("geometric_properties_line", {}).get("xmin_line", float('inf')))) #
        
        for idx, line_obj_final in enumerate(final_reconstructed_lines): #
            line_obj_final['line_id'] = f"final_line_{idx:04d}" #
            for const_el in line_obj_final.get('constituent_elements_ocr_data', []): #
                const_el.pop('shapely_polygon', None) #
                const_el.pop('original_ocr_data', None) #

        logger.info(f"Reconstrucción de líneas finalizada. Total líneas finales: {len(final_reconstructed_lines)}")
        return final_reconstructed_lines
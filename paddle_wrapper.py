# PerfectOCR/core/ocr/paddle_wrapper.py
import os
import cv2 # Aunque PaddleOCR puede tomar rutas, a menudo se le pasa np.ndarray
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union # Union para tipos de coordenadas
from paddleocr import PaddleOCR # Importación principal

logger = logging.getLogger(__name__)

class PaddleOCRWrapper:
    def __init__(self, config_dict: Dict, project_root: str):
        self.paddle_config = config_dict
        self.project_root = project_root # Necesario para resolver rutas relativas de modelos
        logger.debug(f"PaddleOCRWrapper inicializando con config: {self.paddle_config}")

        def resolve_model_path(model_key_in_config: str) -> Optional[str]:
            relative_path = self.paddle_config.get(model_key_in_config)
            if relative_path:
                if os.path.isabs(relative_path):
                    return relative_path if os.path.exists(relative_path) else None
                # Resolver ruta relativa al project_root
                absolute_path = os.path.abspath(os.path.join(self.project_root, relative_path))
                if os.path.exists(absolute_path):
                    logger.debug(f"Modelo {model_key_in_config} encontrado en ruta absoluta resuelta: {absolute_path}")
                    return absolute_path
                else:
                    logger.warning(f"Modelo {model_key_in_config} no encontrado en {absolute_path} (relativo a project_root '{self.project_root}'). "
                                   f"Verifique la ruta '{relative_path}'. PaddleOCR podría intentar descargar o usar defaults.")
                    return relative_path # Devolver la ruta original; PaddleOCR manejará si es inválida o necesita descarga
            return None 

        use_angle_cls_param = self.paddle_config.get('use_angle_cls', True)
        lang_param = self.paddle_config.get('lang', 'es') # Default a español si no se especifica
        show_log_param = self.paddle_config.get('show_log', False) 
        use_gpu_param = self.paddle_config.get('use_gpu', False)
        
        # Resolver rutas de modelos
        det_model_dir_param = resolve_model_path('det_model_dir')
        rec_model_dir_param = resolve_model_path('rec_model_dir')
        cls_model_dir_param = resolve_model_path('cls_model_dir')
        
        # Parámetros adicionales de PaddleOCR que podrían estar en config
        # Ejemplo: det_db_thresh, rec_batch_num, etc.
        # Se pasarían como **kwargs si están definidos en paddle_config
        extra_paddle_params = {
            k: v for k, v in self.paddle_config.items() 
            if k not in ['use_angle_cls', 'lang', 'det_model_dir', 'rec_model_dir', 'cls_model_dir', 'use_gpu', 'show_log']
        }

        try:
            self.engine = PaddleOCR(
                use_angle_cls=use_angle_cls_param,
                lang=lang_param,
                det_model_dir=det_model_dir_param,
                rec_model_dir=rec_model_dir_param,
                cls_model_dir=cls_model_dir_param,
                use_gpu=use_gpu_param,
                show_log=show_log_param,
                **extra_paddle_params # Pasar parámetros adicionales
            )
            #logger.info(f"PaddleOCR engine inicializado exitosamente para idioma '{lang_param}'. GPU: {use_gpu_param}.")
        except Exception as e:
         #   logger.error(f"Error crítico al inicializar PaddleOCR engine: {e}", exc_info=True)
            self.engine = None # Marcar 

    def _parse_paddle_result_to_spec(self, paddle_ocr_result_raw: Optional[List[Any]]) -> List[Dict[str, Any]]:
        output_lines: List[Dict[str, Any]] = []
        if not paddle_ocr_result_raw or paddle_ocr_result_raw[0] is None:
            if paddle_ocr_result_raw is None: # engine.ocr puede devolver None
                logger.info("PaddleOCR no devolvió resultados (None).")
            else:
                logger.info("PaddleOCR no devolvió detecciones válidas para parsear (resultado vacío o primer elemento None).")
            return output_lines

        # PaddleOCR devuelve una lista de listas. La lista externa es por imagen (aquí siempre 1 imagen).
        # La lista interna contiene los resultados para esa imagen.
        items_for_first_image = paddle_ocr_result_raw[0] 
        if items_for_first_image is None: # Puede ser que para la imagen no haya detecciones
            logger.info("PaddleOCR no encontró texto en la imagen provista.")
            return output_lines

        line_counter = 0
        for item_tuple in items_for_first_image: 
            # Cada item_tuple es: (coordenadas_poligono, (texto, confianza))
            # Ejemplo: ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ('texto reconocido', 0.95))
            if not (isinstance(item_tuple, (list, tuple)) and len(item_tuple) == 2):
                logger.warning(f"Formato de item inesperado de PaddleOCR: {item_tuple}")
                continue

            bbox_polygon_raw = item_tuple[0] 
            text_and_confidence = item_tuple[1]

            if not (isinstance(text_and_confidence, (list, tuple)) and len(text_and_confidence) == 2):
                logger.warning(f"Formato de texto/confianza inesperado de PaddleOCR: {text_and_confidence}")
                continue
                
            text, confidence_raw = text_and_confidence[0], text_and_confidence[1]

            polygon_coords_formatted: List[List[float]] = []
            if isinstance(bbox_polygon_raw, list) and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in bbox_polygon_raw):
                try:
                    polygon_coords_formatted = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                except (TypeError, ValueError, IndexError) as e_coord:
                    logger.warning(f"Coordenadas de polígono inválidas en item de PaddleOCR: {bbox_polygon_raw}. Error: {e_coord}. Omitiendo item.")
                    continue
            else:
                logger.warning(f"Formato de coordenadas de polígono inesperado de PaddleOCR: {bbox_polygon_raw}. Omitiendo item.")
                continue 
            
            if not polygon_coords_formatted or len(polygon_coords_formatted) < 3 : # Necesita al menos 3 puntos para un polígono
                logger.warning(f"Coordenadas insuficientes o inválidas para polígono después de formatear: {polygon_coords_formatted}. Omitiendo item.")
                continue

            line_counter += 1
            output_lines.append({
                "line_number": line_counter, 
                "text": str(text).strip(), # Asegurar que sea string y sin espacios extra
                "polygon_coords": polygon_coords_formatted, 
                "confidence": round(float(confidence_raw) * 100.0, 2) if isinstance(confidence_raw, (float, int)) else 0.0
            })
        return output_lines

    def extract_detailed_line_data(self, image: np.ndarray, image_file_name: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        if self.engine is None:
            logger.error("Motor PaddleOCR no inicializado. No se puede extraer texto.")
            processing_time = time.perf_counter() - start_time
            img_h, img_w = image.shape[:2] if image is not None else (0,0)
            return {
                "ocr_engine": "paddleocr", "processing_time_seconds": round(processing_time, 3),
                "image_info": {"file_name": image_file_name, "image_dimensions": {"width": img_w, "height": img_h}},
                "recognized_text": {"full_text": "", "lines": []}, "error": "PaddleOCR engine not initialized"
            }

        img_h, img_w = image.shape[:2] if image is not None else (0,0)
        parsed_lines_result: List[Dict[str, Any]] = []
        reconstructed_full_text = ""
        overall_avg_confidence = 0.0

        try:
            if image is None:
                raise ValueError("La imagen de entrada para PaddleOCR es None.")
            
            # El motor PaddleOCR espera una imagen en formato BGR (si es a color)
            # Si la imagen ya está en escala de grises, PaddleOCR la maneja.
            # Si es BGRA, convertir a BGR.
            if len(image.shape) == 3 and image.shape[2] == 4: # BGRA
                image_for_paddle = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                image_for_paddle = image

            raw_paddle_result_list = self.engine.ocr(image_for_paddle, cls=self.paddle_config.get('use_angle_cls', True))
            
            parsed_lines_result = self._parse_paddle_result_to_spec(raw_paddle_result_list)
            
            if parsed_lines_result: 
                reconstructed_full_text = "\n".join([line['text'] for line in parsed_lines_result if 'text' in line]).strip()
                line_confidences = [line.get('confidence', 0.0) for line in parsed_lines_result if isinstance(line.get('confidence'), (float, int))]
                if line_confidences:
                    overall_avg_confidence = round(float(np.mean(line_confidences)), 2) # Asegurar float
            else:
                logger.info(f"PaddleOCR: No se parsearon líneas para {image_file_name}.")

        except Exception as e:
            logger.error(f"Error en PaddleOCR extract_detailed_line_data para {image_file_name}: {e}", exc_info=True)
            processing_time = time.perf_counter() - start_time
            return {
                "ocr_engine": "paddleocr", "processing_time_seconds": round(processing_time, 3),
                "image_info": {"file_name": image_file_name, "image_dimensions": {"width": img_w, "height": img_h}},
                "recognized_text": {"full_text": "", "lines": []}, "error": str(e)
            }
        
        processing_time = time.perf_counter() - start_time
        logger.info(f"PaddleOCR para '{image_file_name}' completado. Líneas: {len(parsed_lines_result)}, Conf. Promedio Líneas: {overall_avg_confidence:.2f}%, Tiempo: {processing_time:.3f}s")
        
        return {
            "ocr_engine": "paddleocr",
            "processing_time_seconds": round(processing_time, 3),
            "image_info": {
                "file_name": image_file_name,
                "image_dimensions": { "width": img_w, "height": img_h }
            },
            "recognized_text": {
                "full_text": reconstructed_full_text, 
                "lines": parsed_lines_result 
            },
            "overall_confidence_avg_lines": overall_avg_confidence
        }

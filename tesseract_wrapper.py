import re
import pytesseract
import numpy as np
import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TesseractOCR:
    def __init__(self, full_ocr_config: Dict):
        if 'tesseract' not in full_ocr_config or not isinstance(full_ocr_config['tesseract'], dict):
            msg = "La sección 'tesseract' es inválida o no se encuentra en la configuración OCR."
            logger.error(msg)
            raise ValueError(msg)
        
        self.config = full_ocr_config['tesseract']
        self.lang = self.config.get('lang', 'spa+eng')
        self.default_psm = self.config.get('psm', {})
        self.default_oem = self.config.get('oem', {})
        self.char_whitelist = self.config.get('tessedit_char_whitelist', {})
        
        tesseract_cmd_path_from_config = self.config.get('cmd_path')
        if tesseract_cmd_path_from_config:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path_from_config
        elif not pytesseract.pytesseract.tesseract_cmd:
            logger.info("cmd_path de Tesseract no especificado... pytesseract intentará encontrarlo.")
        # logger.debug(f"Tesseract cmd_path: {pytesseract.pytesseract.tesseract_cmd}")

    def _limpiar(self, texto: str) -> str:
        if not isinstance(texto, str): return ""
        texto = texto.replace('\r', ' ').replace('\n', ' ')
        texto = re.sub(r'\|--', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def _build_config_str(self, psm: Optional[int] = None, oem: Optional[int] = None, tessedit_char_whitelist: Optional[str] = None) -> str:
        current_psm = psm if psm is not None else self.default_psm
        current_oem = oem if oem is not None else self.default_oem
        
        config_parts = [f"--psm {current_psm}", f"--oem {current_oem}"]
        
        current_lang = self.config.get('lang', self.lang) 
        config_parts.append(f"-l {current_lang}")
        
        # Whitelist: primero el argumento, luego config, luego nada

        final_char_whitelist = self.config.get("tessedit_char_whitelist")

        if final_char_whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={final_char_whitelist}")
        
        user_words_path = self.config.get("user_words")
        if user_words_path:

            if os.path.exists(str(user_words_path)):
                config_parts.append(f"-c user_words_file=\"{str(user_words_path)}\"")
            else:
                logger.warning(f"Archivo user_words_file no encontrado en: {user_words_path}")
        
        if self.config.get("preserve_interword_spaces") is not None: # Puede ser 0 o 1
            config_parts.append(f"-c preserve_interword_spaces={self.config.get('preserve_interword_spaces')}")

        # Añadir otros parámetros tessedit_* de la configuración
        for key, value in self.config.items():
            if key.startswith("tessedit_") and key not in ["tessedit_char_whitelist", "user_words_file"]:
                config_parts.append(f"-c {key}={value}")
        
        config_str = " ".join(config_parts)
        return config_str

    def extract_detailed_word_data(self, image: np.ndarray, image_file_name: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        config_str = self._build_config_str()
        confidence_threshold = self.config.get('confidence_threshold', 0.5)
        output_words: List[Dict[str, Any]] = []
        text_layout_lines: List[Dict[str, Any]] = []
        img_h, img_w = image.shape[:2] if image is not None else (0,0)
        total_confidence = 0.0
        word_count_for_avg = 0
        
        lines_data_temp: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {} 

        try:
            if image is None:
                raise ValueError("La imagen de entrada para Tesseract es None.")

            data = pytesseract.image_to_data(image, config=config_str, output_type=pytesseract.Output.DICT)
            num_items = len(data['text']) 
            word_internal_counter = 0 

            for i in range(num_items):
                confidence_val_str = data['conf'][i]
                text_val = data['text'][i]

                # Solo procesar elementos que son palabras reales con confianza válida
                if text_val and text_val.strip():
                    try:
                        confidence_val = float(confidence_val_str)
                        if confidence_val < 0 or confidence_val > confidence_threshold:
                            continue
                    except ValueError:
                        logger.warning(f"Valor de confianza no numérico '{confidence_val_str}' para texto '{text_val}'. Omitiendo.")
                        continue

                    txt = self._limpiar(text_val)
                    if not txt: continue

                    word_internal_counter += 1
                    x = int(data['left'][i])
                    y = int(data['top'][i])
                    w = int(data['width'][i])
                    h = int(data['height'][i])

                    if w <= 0 or h <= 0: continue # Omitir geometrías inválidas

                    word_data = {
                        "word_number": word_internal_counter,
                        "text": txt,
                        "bbox": [x, y, x + w, y + h],
                        "confidence": round(confidence_val, 2)
                    }
                    output_words.append(word_data)
                    
                    if confidence_val > 0: # Solo considerar confianzas positivas para el promedio
                        total_confidence += confidence_val
                        word_count_for_avg += 1

                    # Agrupar por línea para text_layout
                    line_key = (int(data['block_num'][i]), int(data['par_num'][i]), int(data['line_num'][i]))
                    if line_key not in lines_data_temp:
                        lines_data_temp[line_key] = []
                    lines_data_temp[line_key].append({"text": txt, "confidence": confidence_val, "x_coord": x})

            # Construir text_layout a partir de lines_data_temp
            inferred_line_num = 0
            for key in sorted(lines_data_temp.keys()): # Ordenar por block, par, line
                words_in_line = sorted(lines_data_temp[key], key=lambda item: item['x_coord']) 
                line_text_parts = [word['text'] for word in words_in_line]
                line_text_str = " ".join(line_text_parts).strip() # Asegurar strip final
                
                if line_text_str: 
                    inferred_line_num += 1
                    line_confidences = [word['confidence'] for word in words_in_line if word['confidence'] >= 0]
                    avg_line_conf = round(sum(line_confidences) / len(line_confidences), 2) if line_confidences else 0.0
                    
                    text_layout_lines.append({
                        "line_number_inferred": inferred_line_num,
                        "line_text": line_text_str,
                        "line_avg_word_confidence": avg_line_conf
                        # Podría añadirse el bbox de la línea si se calcula (min/max de x/y de sus palabras)
                    })

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract ejecutable no encontrado o no configurado correctamente.")
            processing_time = time.perf_counter() - start_time
            return {
                "ocr_engine": "tesseract", "processing_time_seconds": round(processing_time, 3),
                "overall_confidence_words": 0.0,
                "image_info": {"file_name": image_file_name, "image_dimensions": {"width": img_w, "height": img_h}},
                "recognized_text": {"text_layout": [], "words": []}, "error": "TesseractNotFoundError"
            }
        except Exception as e:
            logger.error(f"Error durante la extracción detallada de Tesseract para '{image_file_name}': {e}", exc_info=True)
            processing_time = time.perf_counter() - start_time
            return {
                "ocr_engine": "tesseract", "processing_time_seconds": round(processing_time, 3),
                "overall_confidence_words": 0.0,
                "image_info": {"file_name": image_file_name, "image_dimensions": {"width": img_w, "height": img_h}},
                "recognized_text": {"text_layout": [], "words": []}, "error": str(e)
            }

        processing_time = time.perf_counter() - start_time
        overall_avg_confidence = round(total_confidence / word_count_for_avg, 2) if word_count_for_avg > 0 else 0.0
        
        logger.info(f"Tesseract OCR para '{image_file_name}' completado. Palabras: {len(output_words)}, Líneas inferidas: {len(text_layout_lines)}, Conf. Promedio Palabras: {overall_avg_confidence:.2f}%, Tiempo: {processing_time:.3f}s")
        return {
            "ocr_engine": "tesseract",
            "processing_time_seconds": round(processing_time, 3),
            "overall_confidence_words": overall_avg_confidence,
            "image_info": {
                "file_name": image_file_name,
                "image_dimensions": {"width": img_w, "height": img_h}
            },
            "recognized_text": {
                "text_layout": text_layout_lines, 
                "words": output_words            
            }
        }

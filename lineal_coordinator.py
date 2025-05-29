import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List 
from core.lineal_finder.lineal_reconstructor_worker import LineReconstructorWorker
from utils.encoders import NumpyEncoder
from utils.output_handlers import JsonOutputHandler, TextOutputHandler

logger = logging.getLogger(__name__)

class LinealFinderCoordinator:
    def __init__(self, config: Dict, project_root: str, workflow_config: Optional[Dict] = None):
        self.project_root = project_root
        self.worker_config = config # Esta es la config específica para LineReconstructorWorker
        self.workflow_config = workflow_config or {} # Asegura que sea un dict para evitar errores con .get()

        # Inicializar los handlers aquí
        # Asumiendo que no necesitan configuración específica o que su config default es {}
        self.json_output_handler = JsonOutputHandler(config={})
        self.text_output_handler = TextOutputHandler() # No parece tomar config en su uso actual

        self._page_dimensions: Dict[str, Any] = {}
        logger.info(f"LinealFinderCoordinator inicializado. Config worker: {list(self.worker_config.keys())}")

    def _update_page_dimensions(self, page_dimensions_from_ocr: Optional[Dict[str, Any]]):
        if page_dimensions_from_ocr and \
           isinstance(page_dimensions_from_ocr.get('width'), (int, float)) and \
           isinstance(page_dimensions_from_ocr.get('height'), (int, float)):
            self._page_dimensions = {
                'width': int(page_dimensions_from_ocr['width']),
                'height': int(page_dimensions_from_ocr['height'])
            }
            logger.debug(f"Dimensiones de página actualizadas en LinealFinderCoordinator a: {self._page_dimensions}")
        else:
            logger.warning(f"Dimensiones de página no válidas o no proporcionadas desde OCR: {page_dimensions_from_ocr}. ")
            if not self._page_dimensions: # Solo establecer a {} si estaba completamente vacío
                self._page_dimensions = {}

    def reconstruct_lines_from_ocr_output(self, ocr_output_json: Dict, page_dimensions: Dict[str, Any],
                                          original_doc_base_name: str, output_dir_for_json: str) -> Dict[str, Any]:
        timestamp_start = datetime.now().isoformat()
        logger.debug(f"LinealFinderCoordinator: 'output_dir_for_json' recibido: {output_dir_for_json}")

        self._update_page_dimensions(page_dimensions)

        # Verificar dimensiones primero y que sean positivas
        if not self._page_dimensions or \
           not isinstance(self._page_dimensions.get('width'), (int, float)) or \
           not isinstance(self._page_dimensions.get('height'), (int, float)) or \
           self._page_dimensions['width'] <= 0 or self._page_dimensions['height'] <= 0:
            error_msg = (f"LinealFinderCoordinator no recibió dimensiones de página válidas o positivas para "
                         f"{original_doc_base_name}. Recibido: {self._page_dimensions}")
            logger.error(error_msg)
            return {
                "document_id": original_doc_base_name, "timestamp_line_reconstruction": timestamp_start,
                "status": "error_line_reconstruction_invalid_dims", "message": error_msg,
                "page_dimensions_used": self._page_dimensions, "ocr_metadata_ref": ocr_output_json.get("metadata", {}),
                "line_processing_parameters": self.worker_config, "reconstructed_fused_lines": [], # Cambiado "lines" a "reconstructed_fused_lines"
                "fused_transcription_txt_path": None, "saved_fused_lines_json_path": None
            }

        ocr_raw_results = ocr_output_json.get("ocr_raw_results", {})
        tesseract_data_words = ocr_raw_results.get("tesseract", {}).get("words", [])
        paddle_data_segments = ocr_raw_results.get("paddleocr", {}).get("lines", [])

        logger.debug(f"Datos Tesseract para fusión: {len(tesseract_data_words)} palabras.")
        logger.debug(f"Datos PaddleOCR para fusión: {len(paddle_data_segments)} segmentos.")

        if not tesseract_data_words and not paddle_data_segments:
            message = "No se encontraron datos de palabras/líneas de Tesseract ni PaddleOCR para fusión."
            logger.warning(message)
            return {
                "document_id": original_doc_base_name, "timestamp_line_reconstruction": timestamp_start,
                "status": "no_ocr_elements_for_fusion", "message": message,
                "page_dimensions_used": self._page_dimensions, "ocr_metadata_ref": ocr_output_json.get("metadata", {}),
                "line_processing_parameters": self.worker_config, "reconstructed_fused_lines": [],
                "fused_transcription_txt_path": None, "saved_fused_lines_json_path": None
            }

        try:
            worker = LineReconstructorWorker(
                page_width=int(self._page_dimensions['width']),
                page_height=int(self._page_dimensions['height']),
                config=self.worker_config
            )
            final_fused_reconstructed_lines = worker.fuse_and_reconstruct_all_lines(
                tesseract_raw_words=tesseract_data_words,
                paddle_raw_segments=paddle_data_segments
            )
            logger.info(f"LinealFinderCoordinator: Worker devolvió {len(final_fused_reconstructed_lines if final_fused_reconstructed_lines is not None else [])} líneas finales.")
        except Exception as e:
            message = f"Error durante instanciación o ejecución de LineReconstructorWorker: {e}."
            logger.error(message, exc_info=True)
            return {
                "document_id": original_doc_base_name, "timestamp_line_reconstruction": timestamp_start,
                "status": "error_worker_execution", "message": message,
                "page_dimensions_used": self._page_dimensions, "ocr_metadata_ref": ocr_output_json.get("metadata", {}),
                "line_processing_parameters": self.worker_config, "reconstructed_fused_lines": [],
                "fused_transcription_txt_path": None, "saved_fused_lines_json_path": None
            }

        plain_text_preview_list = []
        if final_fused_reconstructed_lines:
            for line_obj in final_fused_reconstructed_lines:
                plain_text_preview_list.append(line_obj.get('text_raw', ''))

        transcription_file_path = None
        if self.workflow_config.get('generate_pure_text_file', True):
            transcription_lines: List[str] = [] # Definir tipo
            if final_fused_reconstructed_lines:
                transcription_lines.append(f"--- Fused and Reconstructed Lines ({len(final_fused_reconstructed_lines)}) ---")
                for line_obj in final_fused_reconstructed_lines:
                    transcription_lines.append(f"{line_obj.get('line_id', 'UNKNOWN_ID')}: {line_obj.get('text_raw', '')}")
            else:
                transcription_lines.append("No fused lines found.")
            transcription_text_content = "\n".join(transcription_lines) + "\n"

            if output_dir_for_json and isinstance(output_dir_for_json, str) and os.path.isdir(output_dir_for_json):
                transcription_file_name = f"{original_doc_base_name}_fused_transcription.txt"
                # AQUÍ SE USA self.text_output_handler
                transcription_file_path = self.text_output_handler.save(
                    text_content=transcription_text_content,
                    output_dir=output_dir_for_json,
                    file_name_with_extension=transcription_file_name
                )
            elif output_dir_for_json:
                logger.warning(f"LinealFinderCoordinator: El directorio de salida '{output_dir_for_json}' no es válido para la transcripción. No se guardará.")
            else:
                logger.warning("LinealFinderCoordinator: No se proporcionó 'output_dir_for_json'. Transcripción fusionada NO se guardará.")

        final_output_payload = {
            "document_id": original_doc_base_name,
            "timestamp_line_reconstruction": timestamp_start,
            "page_dimensions_used": self._page_dimensions,
            "ocr_metadata_ref": ocr_output_json.get("metadata", {}),
            "line_processing_parameters": self.worker_config,
            "full_transcription_preview": plain_text_preview_list,
            "reconstructed_fused_lines": final_fused_reconstructed_lines,
            "status": "success_fused" if final_fused_reconstructed_lines else "warning_no_fused_lines_generated",
            "fused_transcription_txt_path": transcription_file_path,
            "saved_fused_lines_json_path": None
        }
        # --- Guardado del Payload Final JSON usando JsonOutputHandler ---
        fused_lines_json_path = None
        if output_dir_for_json and isinstance(output_dir_for_json, str) and os.path.isdir(output_dir_for_json):
            json_file_name = f"{original_doc_base_name}_fused_reconstructed_lines.json"
            fused_lines_json_path = self.json_output_handler.save(
                data=final_output_payload,
                output_dir=output_dir_for_json,
                file_name_with_extension=json_file_name
            )
            if fused_lines_json_path:
                final_output_payload["saved_fused_lines_json_path"] = fused_lines_json_path
        elif output_dir_for_json: 
            logger.warning(f"LinealFinderCoordinator: El directorio de salida '{output_dir_for_json}' no es válido para el JSON de líneas fusionadas. No se guardará.")
        else: 
            logger.warning("LinealFinderCoordinator: No se proporcionó 'output_dir_for_json'. JSON de líneas fusionadas NO se guardará.")

        logger.debug(f"Payload final de LinealFinderCoordinator (fusión): {json.dumps(final_output_payload, cls=NumpyEncoder, indent=2)[:500]}...") # Usa NumpyEncoder
        logger.info(f"Proceso de fusión y reconstrucción de líneas para '{original_doc_base_name}' finalizado. "
                    f"Total líneas finales: {len(final_fused_reconstructed_lines if final_fused_reconstructed_lines is not None else [])}.")
        
        return final_output_payload
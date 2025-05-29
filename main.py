# PerfectOCR/main.py
import json
import yaml
import os
import sys
import argparse
import logging
import cv2
import numpy as np
import time
from typing import Dict, Optional, Any, Tuple, List
from coordinators.input_validation_coordinator import InputValidationCoordinator
from coordinators.preprocessing_coordinator import PreprocessingCoordinator
from coordinators.spatial_coordinator import SpatialAnalyzerCoordinator
from coordinators.ocr_coordinator import OCREngineCoordinator
from coordinators.lineal_coordinator import LinealFinderCoordinator
from coordinators.table_field_coordinator import TableAndFieldCoordinator
from utils.output_handlers import JsonOutputHandler
from utils.table_formatter import TableFormatter
from utils.encoders import NumpyEncoder

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MASTER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')

def setup_logging():
    """Configura el sistema de logging centralizado."""
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    if logger_root.hasHandlers():
        logger_root.handlers.clear()

    formatters = {
        'file': logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(module)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ),
        'console': logging.Formatter('%(levelname)s:%(name)s:%(lineno)d - %(message)s')
    }

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatters['file'])
    file_handler.setLevel(logging.DEBUG)
    logger_root.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatters['console'])
    console_handler.setLevel(logging.INFO)
    logger_root.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logging()

class PerfectOCRWorkflow:
    def __init__(self, master_config_path: str):
        self.config = self._load_config(master_config_path)
        self.project_root = PROJECT_ROOT

        # 1. Extraer todas las secciones de configuración primero
        self._extract_config_sections()

        # 2. Ahora que self.workflow_config está definido, puedes usarlo
        self.aggregated_tables_default_name = self.workflow_config.get('aggregated_tables_default_name', "all_tables_summary.txt")

        self._input_validation_coordinator: Optional[InputValidationCoordinator] = None
        self._preprocessing_coordinator: Optional[PreprocessingCoordinator] = None
        self._spatial_analyzer_coordinator: Optional[SpatialAnalyzerCoordinator] = None
        self._ocr_coordinator: Optional[OCREngineCoordinator] = None
        self._lineal_coordinator: Optional[LinealFinderCoordinator] = None
        self._table_extraction_coordinator: Optional[TableAndFieldCoordinator] = None
        self.json_output_handler = JsonOutputHandler(
            config=self.config.get('json_output_handler_config', {})
        )
        logger.debug("PerfectOCRWorkflow listo para inicialización bajo demanda de coordinadores.")

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Configuración maestra cargada desde: {config_path}")
            return config
        except Exception as e:
            logger.critical(f"Error crítico cargando/parseando config maestra {config_path}: {e}", exc_info=True)
            raise

    def _extract_config_sections(self):
        self.workflow_config = self.config.get('workflow', {})
        self.image_preparation_config = self.config.get('image_preparation', {})
        self.spatial_analyzer_config = self.config.get('spatial_analyzer', {})
        self.ocr_config = self.config.get('ocr', {})
        self.line_reconstructor_params_config = self.config.get('line_reconstructor_params', {})
        self.element_fusion_params_config = self.config.get('element_fusion_params', {})
        self.paddle_tesseract_assignment_params_config = self.config.get('paddle_tesseract_assignment_params', {})
        self.table_field_config = self.config.get('table_extraction_config', {})

    @property
    def input_validation_coordinator(self) -> InputValidationCoordinator:
        if self._input_validation_coordinator is None:
            logger.debug("Inicializando InputValidationCoordinator bajo demanda...")
            self._input_validation_coordinator = InputValidationCoordinator(
                config=self.image_preparation_config,
                project_root=self.project_root
            )
        return self._input_validation_coordinator

    @property
    def preprocessing_coordinator(self) -> PreprocessingCoordinator:
        if self._preprocessing_coordinator is None:
            logger.debug("Inicializando PreprocessingCoordinator bajo demanda...")
            self._preprocessing_coordinator = PreprocessingCoordinator(
                config=self.image_preparation_config,
                project_root=self.project_root)
        return self._preprocessing_coordinator

    @property
    def spatial_analyzer_coordinator(self) -> SpatialAnalyzerCoordinator:
        if self._spatial_analyzer_coordinator is None:
            logger.debug("Inicializando SpatialAnalyzerCoordinator bajo demanda...")
            self._spatial_analyzer_coordinator = SpatialAnalyzerCoordinator(
                config=self.spatial_analyzer_config, project_root=self.project_root)
        return self._spatial_analyzer_coordinator

    @property
    def ocr_coordinator(self) -> OCREngineCoordinator:
        if self._ocr_coordinator is None:
            logger.info("Inicializando OCREngineCoordinator bajo demanda (puede tardar por carga de modelos)...")
            self._ocr_coordinator = OCREngineCoordinator(
                config=self.ocr_config, project_root=self.project_root)
        return self._ocr_coordinator

    @property
    def lineal_coordinator(self) -> LinealFinderCoordinator:
        if self._lineal_coordinator is None:
            logger.debug("Inicializando LinealFinderCoordinator bajo demanda...")
            config_for_lineal_finder = {
                "line_reconstructor_params": self.line_reconstructor_params_config,
                "element_fusion_params": self.element_fusion_params_config,
                "paddle_tesseract_assignment_params": self.paddle_tesseract_assignment_params_config
            }
            self._lineal_coordinator = LinealFinderCoordinator(
                config=config_for_lineal_finder,
                project_root=self.project_root,
                workflow_config=self.workflow_config
            )
        return self._lineal_coordinator

    @property
    def table_extraction_coordinator(self) -> TableAndFieldCoordinator:
        if self._table_extraction_coordinator is None:
            logger.info("Inicializando TableAndFieldCoordinator bajo demanda...")
            self._table_extraction_coordinator = TableAndFieldCoordinator(
                config=self.table_field_config,
                project_root=self.project_root
            )
        return self._table_extraction_coordinator

    def _validate_input(self, input_path: str, filename: str) -> Tuple[Optional[Dict[str,Any]], Optional[List[str]], Optional[np.ndarray], float]:
        stage_start_time = time.perf_counter()
        logger.info(f"FASE 1 (método _validate_input): Validando calidad para {filename}")
        quality_assessment, image, time_taken = self.input_validation_coordinator.validate_and_assess_image(input_path)
        elapsed_time = time.perf_counter() - stage_start_time
        logger.info(f"Tiempo de Fase 1 (Validación de Entrada - wrapper) para {filename}: {elapsed_time:.4f} segundos")
        if image is None:
            error_msg = (quality_assessment.get('error', "Fallo en validación/carga")
                        if isinstance(quality_assessment, dict) else "Fallo en validación/carga")
            logger.error(f"Error en validación para {filename}: {error_msg}")
        return quality_assessment, image, time_taken

    def _preprocess_image(self, image: np.ndarray, image_path: str, quality_assessment: Optional[Dict[str,Any]]) -> Tuple[Optional[dict], float]:
        stage_start_time = time.perf_counter()
        logger.info(f"FASE 2 (método _preprocess_image): Preprocesando imagen {os.path.basename(image_path)}")

        # La versión de main.py usa apply_preprocessing_pipeline que toma image_array y quality_assessment_metrics
        results, time_taken = self.preprocessing_coordinator.apply_preprocessing_pipeline(
            image_array=image,
            quality_assessment_metrics=quality_assessment,
            image_path_for_log=image_path
        )
        elapsed_time = time.perf_counter() - stage_start_time
        logger.info(f"Tiempo de Fase 2 (Preprocesamiento - wrapper) para {os.path.basename(image_path)}: {elapsed_time:.4f} segundos")
        return results, time_taken

    def _analyze_spatial_features(self, binary_image: np.ndarray, base_name: str, output_dir: str) -> Tuple[dict, float]:
        stage_start_time = time.perf_counter()
        logger.info(f"FASE 2.5: Iniciando análisis espacial para {base_name}...")
        results = self.spatial_analyzer_coordinator.analyze_image(binary_image)
        elapsed_time = time.perf_counter() - stage_start_time
        logger.info(f"Tiempo de Fase 2.5 (Análisis Espacial) para {base_name}: {elapsed_time:.4f} segundos")
        if results.get("density_map") is not None and output_dir:
            try:
                density_map_path = os.path.join(output_dir, f"{base_name}_density_map.png")
                _ensure_dir_exists(density_map_path)
                density_map_to_save = results["density_map"]
                if np.max(density_map_to_save) <= 1.0 and np.max(density_map_to_save) > 0 :
                    density_map_to_save = (density_map_to_save * 255)
                cv2.imwrite(density_map_path, density_map_to_save.astype(np.uint8))
                logger.debug(f"Mapa de densidad guardado en: {density_map_path}")
            except Exception as e_save_density:
                logger.error(f"Error guardando mapa de densidad para {base_name}: {e_save_density}")
        return results, elapsed_time

    def _run_ocr(self, binary_image: np.ndarray, filename: str, quality_assessment: Optional[Dict[str,Any]]) -> Tuple[Optional[dict], float]:
        stage_start_time = time.perf_counter()
        logger.info(f"FASE 3: Ejecutando OCR para {filename}")
        pil_mode = quality_assessment.get("pil_mode_from_array", quality_assessment.get("pil_mode", "unknown")) \
            if isinstance(quality_assessment, dict) else "unknown"

        results = self.ocr_coordinator.run_ocr_parallel(
            binary_image_for_ocr=binary_image, image_file_name=filename,
            image_pil_mode=pil_mode
        )
        elapsed_time = time.perf_counter() - stage_start_time
        logger.info(f"Tiempo de Fase 3 (Coordinación OCR) para {filename}: {elapsed_time:.4f} segundos")
        if results and results.get("metadata", {}).get("processing_time_seconds"):
            for engine, t_ocr in results["metadata"]["processing_time_seconds"].items():
                logger.info(f"-> Tiempo de motor OCR ({engine}): {t_ocr:.4f} segundos")
        return results, elapsed_time

    def _validate_ocr_results(self, ocr_results: Optional[dict], filename: str) -> bool:
        if not isinstance(ocr_results, dict): return False
        has_tesseract_words = bool(ocr_results.get("ocr_raw_results", {}).get("tesseract", {}).get("words"))
        has_paddle_lines = bool(ocr_results.get("ocr_raw_results", {}).get("paddleocr", {}).get("lines"))
        if not (has_tesseract_words or has_paddle_lines):
            logger.error(f"OCR no produjo texto utilizable para {filename}.")
            return False
        return True

    def _save_ocr_results(self, ocr_results: dict, base_name: str, output_dir: str) -> Optional[str]:
        ocr_json_filename = f"{base_name}_ocr_raw_results.json"
        return self.json_output_handler.save(
            data=ocr_results, output_dir=output_dir, file_name_with_extension=ocr_json_filename
        )

    def _reconstruct_lines(self, ocr_results: dict, base_name: str, output_dir: str) -> Tuple[Optional[dict], float]:
        stage_start_time = time.perf_counter()
        logger.info(f"FASE 4: Reconstruyendo líneas para {base_name}")

        page_dimensions = ocr_results.get("metadata", {}).get("dimensions")
        if not page_dimensions or not page_dimensions.get('width') or not page_dimensions.get('height'):
            logger.error(f"No se pudieron obtener dimensiones de página válidas de ocr_results para {base_name}."
                        f" Dimensiones recibidas: {page_dimensions}. Abortando reconstrucción de líneas.")
            
            return self._build_error_response("error_line_reconstruction", base_name,
                                          "Dimensiones de página no disponibles o inválidas desde OCR.",
                                          "line_reconstruction_setup"), 0.0 # Devolver tiempo 0

        results = self.lineal_coordinator.reconstruct_lines_from_ocr_output(
            ocr_output_json=ocr_results,
            page_dimensions=page_dimensions, #Pasar las dimensiones obtenidas
            original_doc_base_name=base_name,
            output_dir_for_json=output_dir
        )
        elapsed_time = time.perf_counter() - stage_start_time
        logger.info(f"Tiempo de Fase 4 (Reconstrucción de Líneas) para {base_name}: {elapsed_time:.4f} segundos")
        return results, elapsed_time

    def _extract_structured_data(self, line_data: dict, spatial_features: dict,
                                base_name: str, output_dir: str,
                                aggregated_tables_txt_path: Optional[str]) -> Tuple[Optional[dict], float]:
        stage_start_time = time.perf_counter()
        logger.info(f"FASE 5: Extrayendo datos estructurados para {base_name}")

        results = self.table_extraction_coordinator.extract_tables_from_reconstructed_lines(
            reconstructed_line_data=line_data, output_dir_for_json=output_dir,
            original_doc_base_name=base_name, spatial_features=spatial_features
        )
        elapsed_time = time.perf_counter() - stage_start_time
        logger.info(f"Tiempo de Fase 5 (Extracción de Datos Estructurados) para {base_name}: {elapsed_time:.4f} segundos")

        if results and aggregated_tables_txt_path:
            _append_table_to_aggregated_txt(results, base_name, aggregated_tables_txt_path)

        return results, elapsed_time

    def process_document(self, input_path: str, output_dir_override: Optional[str] = None,
                        aggregated_tables_txt_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        overall_start_time = time.perf_counter()
        processing_times_summary: Dict[str, float] = {}
        original_file_name = os.path.basename(input_path)
        base_name = os.path.splitext(original_file_name)[0]

        current_output_dir = output_dir_override if output_dir_override else self.workflow_config.get(
            'output_folder', os.path.join(self.project_root, 'data', 'output_cli_default')
        )
        try:
            os.makedirs(current_output_dir, exist_ok=True)
        except Exception as e_mkdir:
            logger.critical(f"No se pudo crear directorio de salida {current_output_dir}: {e_mkdir}", exc_info=True)
            return self._build_error_response("error_creating_output_dir", original_file_name, str(e_mkdir), "setup")

        # FASE 1: VALIDACIÓN
        logger.info(f"FASE 1: Validando y obteniendo métricas para {original_file_name}")
        quality_metrics, quality_observations, image_loaded_by_validator, time_val = \
            self.input_validation_coordinator.validate_and_assess_image(input_path) #
        processing_times_summary["1_input_validation"] = round(time_val, 4)

        validation_failed = False
        error_msg_val = " "
        if image_loaded_by_validator is None:
            validation_failed = True
            error_msg_val = "Fallo en carga de imagen por InputValidationCoordinator."
        if quality_metrics and quality_metrics.get('error'):
            validation_failed = True
            error_msg_val = quality_metrics.get('error', "Error desconocido en validación de calidad.")
        elif image_loaded_by_validator is None and not error_msg_val.strip():
            validation_failed = True
            error_msg_val = "Fallo en validación/carga de imagen (detalle no especificado en quality_metrics)."

        if validation_failed:
            logger.error(f"Error en validación para {original_file_name}: {error_msg_val}")
            if quality_observations:
                logger.info(f"Observaciones de calidad (durante fallo de validación) para {original_file_name}: {quality_observations}")
            return self._build_error_response("error_input_validation", original_file_name, error_msg_val, "input_validation")

        #logger.info(f"Métricas de calidad para {original_file_name}: {json.dumps(quality_metrics, cls=NumpyEncoder, indent=2)}") #
        #if quality_observations: # Solo loguear si hay observaciones
         #   logger.info(f"Observaciones de calidad para {original_file_name}: {quality_observations}")


        # FASE 2: PREPROCESAMIENTO
        logger.info(f"FASE 2: Preprocesando imagen {original_file_name}")
        preproc_output_tuple = self.preprocessing_coordinator.apply_preprocessing_pipeline( #
            image_array=image_loaded_by_validator,
            quality_assessment_metrics=quality_metrics,
            image_path_for_log=input_path
        )
        if isinstance(preproc_output_tuple, tuple) and len(preproc_output_tuple) == 2:
            preproc_results, time_prep = preproc_output_tuple
        else:
            logger.error(f"Salida inesperada de apply_preprocessing_pipeline para {original_file_name}.")
            preproc_results = None
            time_prep = 0.0

        processing_times_summary["2_preprocessing"] = round(time_prep, 4)

        if not isinstance(preproc_results, dict) or "binary_image_for_ocr" not in preproc_results:
            error_message_prep = "Fallo en preprocesamiento o imagen binaria no generada."
            if isinstance(preproc_results, dict) and preproc_results.get("error"):
                error_message_prep = preproc_results.get("error")
            logger.error(f"Error en preprocesamiento para {original_file_name}: {error_message_prep}")
            return self._build_error_response("error_preprocessing", original_file_name, error_message_prep, "preprocessing")

        binary_image = preproc_results["binary_image_for_ocr"]
        #if preproc_results.get("preprocessing_parameters_used"):
         #   logger.info(f"Parámetros de preprocesamiento usados para {original_file_name}: {json.dumps(preproc_results['preprocessing_parameters_used'], indent=2)}")

        # FASE 2.5: ANÁLISIS ESPACIAL
        spatial_results, time_spatial = self._analyze_spatial_features(binary_image, base_name, current_output_dir)
        processing_times_summary["2.5_spatial_analysis"] = round(time_spatial, 4)

        # FASE 3: OCR
        # Pasamos quality_metrics (que es el quality_assessment de la fase 1)
        ocr_results, time_ocr_coord = self._run_ocr(binary_image, original_file_name, quality_metrics)
        processing_times_summary["3_ocr_coordination"] = round(time_ocr_coord, 4)
        if isinstance(ocr_results, dict) and ocr_results.get("metadata", {}).get("processing_time_seconds"):
            for k, v_ocr in ocr_results["metadata"]["processing_time_seconds"].items():
                processing_times_summary[f"3.1_ocr_engine_{k}"] = round(v_ocr,4)

        if not self._validate_ocr_results(ocr_results, original_file_name):
            return self._build_error_response("error_ocr", original_file_name, "OCR no produjo elementos de texto utilizables.", "ocr")

        ocr_json_path = self._save_ocr_results(ocr_results if ocr_results else {}, base_name, current_output_dir)


        # FASE 4: RECONSTRUCCIÓN DE LÍNEAS
        line_reconstruction, time_line = self._reconstruct_lines(ocr_results if ocr_results else {}, base_name, current_output_dir)
        processing_times_summary["4_line_reconstruction"] = round(time_line, 4)

        if not isinstance(line_reconstruction, dict) or not line_reconstruction.get('status', '').startswith('success'):
            msg_line = line_reconstruction.get('message', 'Fallo en reconstrucción de líneas') if isinstance(line_reconstruction, dict) else 'Fallo en reconstrucción de líneas'
            return self._build_error_response("error_line_reconstruction", original_file_name, msg_line, "line_reconstruction")

        # FASE 5: EXTRACCIÓN DE DATOS ESTRUCTURADOS
        structured_results, time_table = self._extract_structured_data(
            line_reconstruction, spatial_results, base_name, current_output_dir, aggregated_tables_txt_path
        )
        processing_times_summary["5_table_extraction"] = round(time_table, 4)
        overall_processing_time = time.perf_counter() - overall_start_time
        processing_times_summary["total_workflow_document"] = round(overall_processing_time, 4)

        #logger.info(f"Resumen de tiempos de procesamiento para {original_file_name}:\n{json.dumps(processing_times_summary, indent=4, cls=NumpyEncoder)}")

        final_payload = self._build_final_response(
            original_file_name, ocr_json_path, line_reconstruction,
            structured_results if isinstance(structured_results, dict) else {"status_table_extraction": "error_no_table_results_or_failure", "table_matrix": {}}
        )

        if final_payload:
            if "summary" not in final_payload: final_payload["summary"] = {}
            final_payload["summary"]["processing_times_seconds"] = processing_times_summary
            if aggregated_tables_txt_path and os.path.exists(aggregated_tables_txt_path):
                if "outputs" not in final_payload: final_payload["outputs"] = {}
                final_payload["outputs"]["aggregated_tables_summary_txt"] = aggregated_tables_txt_path

        return final_payload

    def _build_error_response(self, status: str, filename: str, message: str, stage: Optional[str] = None) -> dict:
        error_details = {"message": message}
        if stage: error_details["stage"] = stage
        return {"document_id": filename, "status_overall_workflow": status, "error_details": error_details }

    def _build_final_response(self, filename: str, ocr_path: Optional[str], line_data_payload: dict, table_data_payload: dict) -> dict:
        table_status = table_data_payload.get("status_table_extraction", "unknown_table_status")
        final_status = "success"

        if not ocr_path: final_status = "error_ocr_save"
        elif not line_data_payload.get('status', '').startswith('success'): final_status = "error_line_reconstruction"
        elif not table_status.startswith('success'): final_status = "partial_success_table_extraction_issues"

        if table_status.startswith("error_"):
            table_error_msg = "Error desconocido en tabla"
            if table_data_payload.get('table_matrix') and isinstance(table_data_payload['table_matrix'], dict):
                table_error_msg = table_data_payload['table_matrix'].get('error', table_status)
            logger.warning(f"Problemas durante extracción de tabla para {filename}: {table_error_msg}")

        outputs = {
            "ocr_raw_json": ocr_path if ocr_path else "Error guardando resultados OCR",
            "fused_lines_json": line_data_payload.get("saved_fused_lines_json_path"),
            "fused_transcription_txt": line_data_payload.get("fused_transcription_txt_path"),
            "table_extraction_json": table_data_payload.get("table_extraction_json_path"),
            "table_markdown_view": table_data_payload.get("markdown_table_file")
        }
        outputs = {k: v for k, v in outputs.items() if v is not None}

        summary_info = {
            "line_reconstruction_status": line_data_payload.get('status'),
            "table_extraction_status": table_status,
        }
        if isinstance(table_data_payload.get("table_matrix"), dict):
            summary_info["table_rows_extracted"] = len(table_data_payload.get("table_matrix", {}).get("rows", []))
            summary_info["table_columns_extracted"] = len(table_data_payload.get("table_matrix", {}).get("headers", []))

        if isinstance(table_data_payload.get("data_rows_info"), dict):
            summary_info.update(table_data_payload.get("data_rows_info"))

        return {"document_id": filename, "status_overall_workflow": final_status, "outputs": outputs, "summary": summary_info}

def _ensure_dir_exists(file_path: str):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory): # Solo crear si el directorio no es vacío y no existe
        try:
            os.makedirs(directory) # No necesita exist_ok=True debido a la condición previa
        except OSError as e:
            logger.error(f"No se pudo crear el directorio {directory}: {e}")

def _append_times_to_txt_file(times_data: Dict, filename: str, output_times_file: Optional[str]):
    if not output_times_file: return
    _ensure_dir_exists(output_times_file)
    try:
        with open(output_times_file, 'a', encoding='utf-8') as f:
            f.write(f"--- Tiempos para: {filename} ---\n")
            for stage, duration in times_data.items():
                f.write(f"{stage}: {duration:.4f} segundos\n")
            f.write("-" * 30 + "\n")
        logger.info(f"Tiempos de procesamiento para {filename} añadidos a {output_times_file}")
    except Exception as e:
        logger.error(f"Error escribiendo tiempos en {output_times_file}: {e}")

def _append_table_to_aggregated_txt(table_payload: Optional[Dict],
                                    original_doc_base_name: str,
                                    aggregated_txt_file_path: Optional[str]):
    if not aggregated_txt_file_path or not table_payload:
        return

    _ensure_dir_exists(aggregated_txt_file_path)
    output_content = f"--- Tabla Extraída para: {original_doc_base_name} ---\n"

    if table_payload.get('status_table_extraction', '').startswith('success'):
        table_matrix_data = table_payload.get('table_matrix', {})
        headers = table_matrix_data.get('headers', []) if isinstance(table_matrix_data.get('headers'), list) else []
        # Asegurarse que las filas sean listas de dicts para TableFormatter
        rows_raw = table_matrix_data.get('rows', []) if isinstance(table_matrix_data.get('rows'), list) else []

        rows_for_formatter = []
        if rows_raw and isinstance(rows_raw[0], list) and isinstance(rows_raw[0][0], dict):
            for row_of_cells in rows_raw:
                rows_for_formatter.append([cell.get("text", "") for cell in row_of_cells])
        else: # Asumir que ya es una lista de listas de strings o algo compatible
            rows_for_formatter = rows_raw

        if headers or rows_for_formatter:
            markdown_table_str = TableFormatter.format_as_markdown(headers, rows_for_formatter)
            output_content += markdown_table_str
        else:
            output_content += "(No se extrajo una tabla o la tabla estaba vacía)\n"
    else:
        output_content += f"Status: {table_payload.get('status_table_extraction')}\n"
        error_details_dict = table_payload.get('table_matrix',{})
        error_msg = "Detalles no disponibles."
        if isinstance(error_details_dict, dict):
            error_msg = error_details_dict.get('error', 'Detalles no disponibles en table_matrix.')
        output_content += f"Mensaje: {error_msg}\n"

    output_content += "\n\n" + "="*80 + "\n\n"

    try:
        with open(aggregated_txt_file_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
        logger.info(f"Información de tabla para '{original_doc_base_name}' añadida a: {aggregated_txt_file_path}")
    except Exception as e:
        logger.error(f"Error añadiendo info de tabla de '{original_doc_base_name}' a '{aggregated_txt_file_path}': {e}")

def _process_single_file(workflow: PerfectOCRWorkflow, file_path: str, output_dir: str,
                        output_times_file: Optional[str], aggregated_tables_txt_path: Optional[str]):
    logger.info(f"Procesando archivo individual: {file_path}")
    result = workflow.process_document(file_path, output_dir, aggregated_tables_txt_path)
    if result:
        times_data = result.get("summary", {}).get("processing_times_seconds")
        if times_data and output_times_file:
            _append_times_to_txt_file(times_data, os.path.basename(file_path), output_times_file)
    else:
        logger.error(f"No se obtuvo resultado para {os.path.basename(file_path)}")

def _process_batch(workflow: PerfectOCRWorkflow, dir_path: str, output_dir: str,
                   output_times_file: Optional[str], aggregated_tables_txt_path: Optional[str]):
    logger.info(f"Procesando directorio en lote: {dir_path}")
    processed_count = 0
    error_count = 0
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
            logger.info(f"Procesando archivo en lote: {filename}")
            try:
                result = workflow.process_document(file_path, output_dir, aggregated_tables_txt_path)
                if result and result.get("summary", {}).get("processing_times_seconds") and output_times_file:
                    _append_times_to_txt_file(result["summary"]["processing_times_seconds"], filename, output_times_file)

                if result and result.get("status_overall_workflow", "").startswith("success"):
                    processed_count += 1
                elif result:
                    error_detail_msg = "Error desconocido"
                    if result.get("error_details") and isinstance(result["error_details"], dict):
                        error_detail_msg = result["error_details"].get('message', 'Error desconocido')
                    elif isinstance(result.get("error_details"), str):
                        error_detail_msg = result.get("error_details")
                    logger.error(f"Error o fallo parcial procesando {filename}: {result.get('status_overall_workflow')} - {error_detail_msg}")
                    error_count +=1
                else:
                    logger.error(f"Procesamiento de {filename} devolvió None o resultado inesperado."); error_count += 1
            except Exception as e:
                logger.error(f"Excepción crítica procesando {filename}: {e}", exc_info=True); error_count += 1
        elif os.path.isfile(file_path):
            logger.info(f"Omitiendo archivo no soportado en lote: {filename}")
    logger.info(f"Procesamiento en lote completado. Éxitos (incl. parciales): {processed_count}, Errores/Fallos: {error_count}")

def main_cli():
    parser = argparse.ArgumentParser(description='PerfectOCR - Procesamiento de Documentos')
    parser.add_argument('input', help='Ruta del archivo o directorio de entrada')
    parser.add_argument('-o', '--output', default=None, help='Directorio de salida (opcional)')
    parser.add_argument('--times_file', default=None, help='Ruta al archivo .txt para guardar los tiempos de procesamiento (opcional)')
    parser.add_argument('--aggregated_tables_file', default=None,
                        help='Ruta al archivo .txt para guardar el resumen de todas las tablas (opcional). Por defecto: [output_dir]/[config_value_or_default_name].txt')
    args = parser.parse_args()
    logger.debug(f"Argumentos recibidos - input: '{args.input}', output: '{args.output}', "
                f"times_file: '{args.times_file}', aggregated_tables_file: '{args.aggregated_tables_file}'")

    try:
        workflow = PerfectOCRWorkflow(master_config_path=MASTER_CONFIG_FILE)
    except Exception as e:
        logger.critical(f"No se pudo inicializar PerfectOCRWorkflow: {e}", exc_info=True); sys.exit(1)

    # Determinar output_dir
    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir_from_config = workflow.workflow_config.get('output_folder')
        if output_dir_from_config:
            if os.path.isabs(output_dir_from_config):
                output_dir = output_dir_from_config
            else: # Si es relativa, se construye desde PROJECT_ROOT
                output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, output_dir_from_config))
        else: # Fallback si no está en args ni en config
            output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'output_cli_default'))
            logger.warning(f"Directorio de salida no especificado ni en argumentos ni en config. Usando por defecto: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Directorio de salida configurado en: {output_dir}")
    except Exception as e:
        logger.critical(f"No se pudo crear directorio de salida {output_dir}: {e}", exc_info=True); sys.exit(1)

    # Determinar output_times_file_path
    output_times_file_path = args.times_file
    if output_times_file_path:
        output_times_file_path = os.path.abspath(output_times_file_path)
        _ensure_dir_exists(output_times_file_path)
        if os.path.exists(output_times_file_path):
            try: open(output_times_file_path, 'w').close(); logger.info(f"Archivo de tiempos '{output_times_file_path}' limpiado.")
            except Exception as e: logger.error(f"No se pudo limpiar '{output_times_file_path}': {e}")
    else:
        output_times_file_path = None
#        logger.info("No se especificó archivo de tiempos (--times_file). No se guardarán los tiempos detallados.")

    # Determinar aggregated_tables_final_path
    # Acceder al nombre de archivo por defecto desde la instancia de workflow
    aggregated_tables_file_name_from_config = workflow.aggregated_tables_default_name

    aggregated_tables_final_path = args.aggregated_tables_file
    if not aggregated_tables_final_path:
        aggregated_tables_final_path = os.path.join(output_dir, aggregated_tables_file_name_from_config)
    else:
        aggregated_tables_final_path = os.path.abspath(aggregated_tables_final_path)
    _ensure_dir_exists(aggregated_tables_final_path)

    if os.path.exists(aggregated_tables_final_path):
        try:
            open(aggregated_tables_final_path, 'w').close()
            #logger.info(f"Archivo de tablas agregadas existente '{aggregated_tables_final_path}' limpiado para nueva sesión.")
        except Exception as e:
            logger.error(f"No se pudo limpiar el archivo de tablas agregadas '{aggregated_tables_final_path}': {e}")

    input_path_abs = os.path.abspath(args.input)
    logger.info(f"Procesando ruta: '{input_path_abs}'")
    if output_times_file_path:
        logger.info(f"Los tiempos de procesamiento se guardarán en: {output_times_file_path}")
    if aggregated_tables_final_path:
        logger.info(f"Las tablas agregadas se guardarán en: {aggregated_tables_final_path}")

    if os.path.isfile(input_path_abs):
        _process_single_file(workflow, input_path_abs, output_dir, output_times_file_path, aggregated_tables_final_path)
    elif os.path.isdir(input_path_abs):
        _process_batch(workflow, input_path_abs, output_dir, output_times_file_path, aggregated_tables_final_path)
    else:
        logger.error(f"Ruta de entrada inválida: '{input_path_abs}'"); sys.exit(1)

if __name__ == "__main__":
    main_cli()
    logger.info("Ejecución completada.")
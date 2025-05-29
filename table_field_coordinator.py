# PerfectOCR/coordinators/table_field_coordinator.py (Versión Refactorizada)
import logging
from typing import Dict, Any, Optional, List # Asegurar todas las importaciones
from datetime import datetime
import os # Necesario para os.path.join

from core.table_extractor.table_data_preparer import TableDataPreparer
from core.table_extractor.table_boundary_detector import TableBoundaryDetector
from core.table_extractor.column_definer import ColumnDefiner
from core.table_extractor.row_definer import RowDefiner
from core.table_extractor.table_row_validator import TableRowValidator
from core.table_extractor.table_cell_assigner import TableCellAssigner
from core.table_extractor.table_cell_post_processor import TableCellPostProcessor
from core.table_extractor.spatial_validator_worker import SpatialValidatorWorker
from core.table_extractor.cell_content_analyzer_worker import CellContentAnalyzerWorker
from utils.geometry_transformers import vectorize_element_list, devectorize_element_list # Añadir devectorize
from utils.output_handlers import JsonOutputHandler, MarkdownOutputHandler
from utils.table_formatter import TableFormatter # Para el Markdown
from utils.encoders import NumpyEncoder # Para guardado JSON complejo

logger = logging.getLogger(__name__)

class TableAndFieldCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.table_config = config 
        self.project_root = project_root
        self.page_dimensions: Dict[str, Any] = {}
        spatial_validator_cfg = self.table_config.get('spatial_validator_worker_config', {})
        spatial_validator_worker_instance = SpatialValidatorWorker(config=spatial_validator_cfg)
        self.data_preparer = TableDataPreparer(
            data_filter_config = self.table_config.get('data_filter_config', {}),
            spatial_validator_worker = spatial_validator_worker_instance
        )

        boundary_detector_cfg = {
            'header_detector_config': self.table_config.get('header_detector_config', {}),
            'spatial_table_end_detection': self.table_config.get('spatial_table_end_detection', {}),
            'table_end_keywords': self.table_config.get('header_detector_config', {}).get('table_end_keywords', [])
        }       

        self.boundary_detector = TableBoundaryDetector(
            config=boundary_detector_cfg, 
            project_root=self.project_root 
        )
        
        self.column_definer = ColumnDefiner(self.table_config.get('column_definer_config', {}))
        self.row_definer = RowDefiner(self.table_config.get('row_definer_config', {}))
        self.row_validator = TableRowValidator(self.table_config.get('row_validation_config', {})) 
        self.cell_assigner = TableCellAssigner(self.table_config.get('cell_assigner_config', {}))
        
        cell_analyzer_cfg = self.table_config.get('cell_content_analyzer_worker_config', {})
        cell_analyzer_worker_instance = CellContentAnalyzerWorker(config=cell_analyzer_cfg)
        
        self.cell_post_processor = TableCellPostProcessor(
            spatial_validator_worker=spatial_validator_worker_instance,
            cell_content_analyzer_worker=cell_analyzer_worker_instance,
            post_processing_rules_config=self.table_config.get('cell_postprocessing_rules_config', {}) 
        )
        
        # output_handlers_config se toma del config general del workflow si es necesario, o default {}
        # No hay una sección 'json_output_handler_config' específica para este coordinador en master_config
        self.json_output_handler = JsonOutputHandler(config=self.table_config.get('json_output_handler_options', {}))
        self.markdown_output_handler = MarkdownOutputHandler(config=self.table_config.get('markdown_output_handler_options', {}))
        logger.info("TableAndFieldCoordinator (refactorizado) inicializado.")

    def _build_error_output(self, 
                            # Se necesita renombrar el parámetro para que no colisione con el módulo `details`
                            ocr_metadata: Optional[Dict[str, Any]],
                            error_message: str,
                            stage: str,
                            details_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Renombrado de 'details' a 'details_info'
        if ocr_metadata is None:
            ocr_metadata = {"warning": "OCR metadata not available for this error report."}
        if details_info is None: # Usar el parámetro renombrado
            details_info = {}
        logger.error(f"TableAndFieldCoordinator: Error en etapa '{stage}' - {error_message}. Detalle: {details_info}")
        return {
            "metadata": ocr_metadata,
            "table_matrix": {"headers": [], "rows": [], "error": error_message, "stage": stage, "details": details_info},
            "status_table_extraction": f"error_{stage.lower().replace(' ', '_')}",
        }

    def extract_tables_from_reconstructed_lines(self,
                                               reconstructed_line_data: Dict[str, Any],
                                               output_dir_for_json: str, # Directorio de salida
                                               original_doc_base_name: str,
                                               spatial_features: Optional[Dict[str, Any]] = None
                                              ) -> Dict[str, Any]:
        
        page_dims_from_input = reconstructed_line_data.get('page_dimensions_used', {})
        if not page_dims_from_input or \
           not isinstance(page_dims_from_input.get('width'), (int, float)) or \
           not isinstance(page_dims_from_input.get('height'), (int, float)) or \
           page_dims_from_input['width'] <= 0 or page_dims_from_input['height'] <= 0:
            error_msg = (f"Dimensiones de página no válidas o faltantes recibidas por TableAndFieldCoordinator: "
                        f"{page_dims_from_input} para {original_doc_base_name}")
            logger.error(error_msg)
            # Pasar ocr_metadata si está disponible, o un dict vacío
            ocr_meta_ref_for_error = reconstructed_line_data.get('ocr_metadata_ref', {})
            return self._build_error_output(ocr_metadata=ocr_meta_ref_for_error, 
                                            error_message=error_msg, 
                                            stage="setup_dimensions", 
                                            details_info={"received_dimensions": page_dims_from_input})
        
        self.page_dimensions = page_dims_from_input # Establecer las dimensiones de la página para la instancia
        
        ocr_metadata = reconstructed_line_data.get('ocr_metadata_ref', {})
        density_map = spatial_features.get("density_map") if spatial_features else None

        all_fused_lines_original = reconstructed_line_data.get('reconstructed_fused_lines', [])
        if not all_fused_lines_original:
            return self._build_error_output(ocr_metadata, "No hay líneas fusionadas reconstruidas.", "input_validation")

        # --- 1. Preparación de Datos ---
        try:
            prepared_data = self.data_preparer.prepare(
                reconstructed_fused_lines=all_fused_lines_original,
                density_map=density_map,
                page_dimensions=self.page_dimensions
            )
            lines_for_boundary_detection = prepared_data.get('header_candidate_lines', [])
            all_valid_words_for_table_vectorized = vectorize_element_list(prepared_data.get('all_valid_words_for_table', [])) #
            if not lines_for_boundary_detection:
                return self._build_error_output(ocr_metadata, "No hay líneas candidatas para detección de límites después de la preparación.", "data_preparation")
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en DataPreparer: {e}", "data_preparation", {"exception": str(e)})
        
        # --- 2. Detección de Límites de Tabla y Encabezados ---
        try:
            self.boundary_detector.set_page_dimensions(self.page_dimensions) 
            boundary_results = self.boundary_detector.detect_boundaries(
                lines_for_search=lines_for_boundary_detection,
                density_map=density_map,
                page_dimensions=self.page_dimensions 
            )

            header_words = boundary_results.get('header_words')
            y_max_header_band = boundary_results.get('y_max_header_band')
            y_min_table_end = boundary_results.get('y_min_table_end')
            if not header_words or y_max_header_band is None or y_min_table_end is None:
                return self._build_error_output(ocr_metadata, "No se identificaron límites de tabla o palabras de encabezado.", "boundary_detection")
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en BoundaryDetector: {e}", "boundary_detection", {"exception": str(e)})

        # --- 3. Definición de Columnas ---
        try:
            # Devectorizar header_words antes de pasarlos a ColumnDefiner si ColumnDefiner espera geometría completa
            header_words_devectorized = devectorize_element_list(header_words)
            defined_columns = self.column_definer.define_table_columns(header_words_devectorized)

            if not defined_columns:
                return self._build_error_output(ocr_metadata, "No se definieron columnas a partir de los encabezados.", "column_definition")
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en ColumnDefiner: {e}", "column_definition", {"exception": str(e)})

        # --- 4. Definición de Filas ---
        try:
            data_rows_unfiltered = self.row_definer.define_data_rows(
                all_document_words=all_valid_words_for_table_vectorized, # Usar los vectorizados
                y_max_header_band=y_max_header_band,
                y_min_table_end=y_min_table_end
            )
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en RowDefiner: {e}", "row_definition", {"exception": str(e)})

        # --- 5. Validación/Limpieza de Filas ---
        try:
            # RowValidator podría necesitar las palabras devectorizadas si opera sobre geometría detallada
            # Asumimos que define_data_rows devuelve filas con palabras vectorizadas
            # y RowValidator puede manejar esto o se devectorizan aquí si es necesario.
            # Por ahora, se asume que RowValidator puede trabajar con la estructura actual.
            data_rows_validated = self.row_validator.validate_rows(data_rows_unfiltered) 
            if not data_rows_validated:
                logger.warning("No quedaron filas de datos después de la validación/limpieza.")
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en RowValidator: {e}", "row_validation", {"exception": str(e)})

        # --- 6. Asignación de Celdas ---
        try:
            # TableCellAssigner espera 'words_in_row' con 'vector_representation'
            table_matrix_cells_assigned_vectorized = self.cell_assigner.assign_words_to_cells(
                data_rows_with_words=data_rows_validated, # Usar filas validadas (con palabras vectorizadas)
                defined_columns=defined_columns # defined_columns ya tiene la geometría de columna
            )
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en TableCellAssigner: {e}", "cell_assignment", {"exception": str(e)})

        # --- 7. Post-Procesamiento de Celdas ---
        try:
            # Devectorizar las celdas antes del post-procesamiento si este último espera geometría completa
            table_matrix_cells_assigned_devectorized = []
            for row_vec in table_matrix_cells_assigned_vectorized:
                devec_row = []
                for cell_vec in row_vec:
                    new_cell = cell_vec.copy()
                    if 'words' in new_cell and isinstance(new_cell['words'], list):
                        new_cell['words'] = devectorize_element_list(new_cell['words'])
                    devec_row.append(new_cell)
                table_matrix_cells_assigned_devectorized.append(devec_row)

            final_table_matrix_cells = self.cell_post_processor.process_cells(
                assigned_table_matrix=table_matrix_cells_assigned_devectorized, # Pasar celdas devectorizadas
                defined_columns=defined_columns,
                data_rows=devectorize_element_list(data_rows_validated), # Devectorizar data_rows también si es necesario para el post-procesador
                density_map=density_map
            )
        except Exception as e:
            return self._build_error_output(ocr_metadata, f"Fallo en TableCellPostProcessor: {e}", "cell_postprocessing", {"exception": str(e)})

        logger.info(f"Extracción de tabla (refactorizada) completada. Columnas: {len(defined_columns)}, Filas: {len(final_table_matrix_cells)}")

        # --- 8. Preparación y Guardado de Resultados ---
        
        # Crear el payload de datos de la tabla para guardar en JSON
        # Este es el payload que quieres que se guarde en *_TABLE_EXTRACTION.json
        table_extraction_output_payload = {
            "table_matrix": final_table_matrix_cells, # Matriz final con texto y palabras (devectorizadas)
            "column_headers_text": [col.get("H_k_text") for col in defined_columns],
            "column_details": defined_columns,
            "source_document_id": original_doc_base_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "page_dimensions_used": self.page_dimensions,
            "ocr_metadata_ref": ocr_metadata, # Incluir referencia a metadatos OCR
            "data_rows_info": {
                "identified_data_rows_count": len(data_rows_validated if data_rows_validated is not None else []),
                "table_end_y_coordinate_used": y_min_table_end
            }
        }
        
        # Guardar el JSON de extracción de tabla
        table_extraction_json_path = None
        if output_dir_for_json and isinstance(output_dir_for_json, str):
            if not os.path.isdir(output_dir_for_json):
                try:
                    os.makedirs(output_dir_for_json, exist_ok=True)
                except OSError as e_mkdir:
                    logger.error(f"No se pudo crear el directorio de salida para el JSON de la tabla: {output_dir_for_json}. Error: {e_mkdir}")
            
            if os.path.isdir(output_dir_for_json):
                table_json_filename = f"{original_doc_base_name}_TABLE_EXTRACTION.json"
                table_extraction_json_path = self.json_output_handler.save(
                    data=table_extraction_output_payload, # Guardar el payload detallado
                    output_dir=output_dir_for_json,
                    file_name_with_extension=table_json_filename
                )
                if not table_extraction_json_path:
                    logger.error(f"No se pudo guardar el archivo JSON de extracción de tabla para {original_doc_base_name}")
            else:
                logger.warning(f"El directorio de salida '{output_dir_for_json}' no es válido o no se pudo crear. El JSON de tabla no se guardará.")
        else:
            logger.warning("No se proporcionó 'output_dir_for_json' o no es un string. El JSON de tabla no se guardará.")

        # Preparar el payload de retorno para el flujo principal de PerfectOCRWorkflow
        # Este payload es más un resumen y contiene las rutas a los archivos guardados.
        return_payload_for_workflow = { 
            "status_table_extraction": "success" if final_table_matrix_cells is not None else "warning_no_table_matrix_generated",
            "ocr_metadata_ref": ocr_metadata, 
            "page_dimensions_used": self.page_dimensions, 
            "table_matrix": { # Resumen para el payload principal
                "headers": [col.get("H_k_text") for col in defined_columns], 
                "rows": final_table_matrix_cells if final_table_matrix_cells is not None else [], # Incluir la matriz aquí también
                "column_details": defined_columns 
            },
            "data_rows_info": { 
                "identified_data_rows_count": len(data_rows_validated if data_rows_validated is not None else []), 
                "table_end_y_coordinate_used": y_min_table_end 
            },
            "table_extraction_json_path": table_extraction_json_path, # <--- AÑADIR LA RUTA DEL ARCHIVO JSON
            "markdown_table_file": None # Esta se generará y añadirá en main.py si es necesario
        }
        
        # Generar y guardar el Markdown para la vista rápida (opcionalmente podría hacerse en main.py también)
        if table_extraction_json_path and final_table_matrix_cells is not None: # Solo si la tabla se extrajo y guardó
            md_filename = f"{original_doc_base_name}_TABLE_VIEW.md"
            md_path = self.markdown_output_handler.save_table_view(
                headers=[col.get("H_k_text") for col in defined_columns],
                table_matrix=final_table_matrix_cells, # Usar la matriz final
                output_dir=output_dir_for_json,
                file_name_with_extension=md_filename,
                document_title=original_doc_base_name
            )
            if md_path:
                return_payload_for_workflow["markdown_table_file"] = md_path


        return return_payload_for_workflow
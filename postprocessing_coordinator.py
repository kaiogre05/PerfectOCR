# PerfectOCR/coordinators/postprocessing_coordinator.py
import logging
import os
from typing import Dict, Any, List, Optional
from core.postprocessing.correctors import TextCorrector
from core.postprocessing.formatters import TextFormatter

logger = logging.getLogger(__name__)

class PostprocessingCoordinator:
    def __init__(self, config: Dict, project_root: str):
        self.config = config
        self.project_root = project_root

        corrector_cfg = self.config.get('text_correction', {})
        # TextCorrector necesita la ruta completa al vocabulario
        # La ruta en config ('vocab_path') debe ser relativa al project_root o absoluta
        # O TextCorrector debe ser adaptado para tomar project_root y resolver la ruta
        self.corrector = TextCorrector(
            # Pasamos directamente la config del corrector y el project_root
            postprocessing_config_path=None, # No es necesario si pasamos la config directamente
            project_root_path=self.project_root,
            correction_config_override=corrector_cfg # Nueva forma de pasar la config
        )

        formatter_cfg = self.config.get('text_formatting', {})
        self.formatter = TextFormatter() # TextFormatter usa métodos estáticos, pero podrías instanciarlo si tuviera config

        logger.info(f"PostprocessingCoordinator inicializado.")

    def correct_and_format(self, text_segments: List[Dict], config_override: Optional[Dict] = None) -> List[Dict]:
        current_config = config_override if config_override else self.config
        logger.debug("PostprocessingCoordinator.correct_and_format llamado")

        # Ejemplo de cómo podrías iterar y aplicar
        processed_segments = []
        for segment in text_segments:
            text_to_process = segment.get('text', '') # Asumiendo que los segmentos tienen una clave 'text'

            if current_config.get('text_correction', {}).get('enabled', True): # Si se habilita la corrección
                text_to_process = self.corrector.fix_common_errors(text_to_process)
                text_to_process = self.corrector.correct_spelling(text_to_process)

            if current_config.get('text_formatting', {}).get('format_dates', False):
                text_to_process = self.formatter.normalize_dates(text_to_process)
            if current_config.get('text_formatting', {}).get('normalize_numbers', False):
                text_to_process = self.formatter.normalize_numbers(text_to_process)

            # Actualizar el segmento con el texto procesado
            # Esto es una simplificación, tu estructura de 'segmento' puede ser más compleja
            # (ej. si procesas palabras dentro de líneas)
            updated_segment = segment.copy()
            if 'text' in updated_segment: # Si es un texto simple
                 updated_segment['text'] = text_to_process
            elif 'line_text_reconstructed_from_tess' in updated_segment : # Si es de la fusión
                # Decidir qué texto procesar y dónde guardarlo
                original_line_text = updated_segment.get('line_text_reconstructed_from_tess', '')
                corrected_line_text = self.corrector.fix_common_errors(original_line_text) # Ejemplo
                corrected_line_text = self.corrector.correct_spelling(corrected_line_text) # Ejemplo
                updated_segment['corrected_line_text'] = corrected_line_text
                # También podrías procesar palabras individuales dentro de 'segment['words']'

            processed_segments.append(updated_segment)

        return processed_segments
# PerfectOCR/core/table_extractor/table_row_validator.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TableRowValidator:
    def __init__(self, row_validation_config: Dict):
        self.config = row_validation_config
        self.min_words = self.config.get('min_words_in_data_row', {})
        self.min_alphanum_chars = self.config.get('min_alphanumeric_chars_in_data_row', {})
        # Se pueden añadir más parámetros de configuración para reglas más complejas
        logger.info(f"TableRowValidator inicializado con config: {self.config}")

    def validate_rows(self, data_rows_defined: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data_rows_defined:
            return []
        
        validated_rows = []
        for idx, row_def in enumerate(data_rows_defined):
            words_in_row = row_def.get('words_in_row', [])
            
            # Regla 1: Mínimo de palabras en la fila
            if len(words_in_row) < self.min_words:
                text_preview = " ".join(w.get('text', '') for w in words_in_row[:3]).strip()
                logger.debug(f"TableRowValidator: Filtrando fila {idx} (pocas palabras: {len(words_in_row)} < {self.min_words}): '{text_preview[:50]}...'")
                continue

            # Regla 2: Mínimo de caracteres alfanuméricos
            text_content = "".join(w.get('text', '') for w in words_in_row).strip()
            alphanum_chars_count = sum(1 for char in text_content if char.isalnum())
            
            if alphanum_chars_count < self.min_alphanum_chars:
                logger.debug(f"TableRowValidator: Filtrando fila {idx} (pocos alfanum: {alphanum_chars_count} < {self.min_alphanum_chars}): '{text_content[:50]}...'")
                continue
            
            # Regla 3: Filtro específico para "t w" (ejemplo, podría generalizarse)
            if len(words_in_row) == 1 and words_in_row[0].get('text', '').strip().lower() == "t w":
                 logger.info(f"TableRowValidator: Filtrando fila {idx} específica de ruido: '{words_in_row[0].get('text')}'")
                 continue
            
            # Si pasa todos los filtros, añadir a las filas validadas
            validated_rows.append(row_def)
        
        logger.info(f"TableRowValidator: {len(data_rows_defined)} filas de entrada -> {len(validated_rows)} filas validadas.")
        return validated_rows
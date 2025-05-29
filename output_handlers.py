# PerfectOCR/utils/output_handlers.py
import json
import os
import logging
from typing import Dict, Any, Optional, List
from .encoders import NumpyEncoder # Asumiendo que NumpyEncoder está en utils.encoders
from utils.table_formatter import TableFormatter
logger = logging.getLogger(__name__)

class JsonOutputHandler:
    """
    Obrero especializado en guardar datos en formato JSON.
    """
    def __init__(self, config: Optional[Dict] = None):
        # La configuración podría usarse para opciones de formato JSON globales si fuera necesario.
        self.config = config if config is not None else {}
        # logger.debug("JsonOutputHandler initialized.")

    def save(self, data: Dict[str, Any], output_dir: str, file_name_with_extension: str) -> Optional[str]:
        """
        Guarda un diccionario de datos en un archivo JSON.
        Crea el directorio de salida si no existe.

        Args:
            data: El diccionario a guardar.
            output_dir: El directorio donde se guardará el archivo.
            file_name_with_extension: El nombre del archivo (e.g., "results.json").

        Returns:
            La ruta completa al archivo guardado si tiene éxito, None en caso contrario.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name_with_extension)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"Datos JSON guardados en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error guardando JSON en {output_path}: {e}", exc_info=True)
            return None
            
class TextOutputHandler:
    """
    Obrero especializado en guardar contenido de texto plano.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        # logger.debug("TextOutputHandler initialized.")

    def save(self, text_content: str, output_dir: str, file_name_with_extension: str) -> Optional[str]:
        """
        Guarda una cadena de texto en un archivo.
        Crea el directorio de salida si no existe.

        Args:
            text_content: La cadena de texto a guardar.
            output_dir: El directorio donde se guardará el archivo.
            file_name_with_extension: El nombre del archivo (e.g., "transcription.txt").

        Returns:
            La ruta completa al archivo guardado si tiene éxito, None en caso contrario.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name_with_extension)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            logger.info(f"Datos de texto guardados en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error guardando archivo de texto en {output_path}: {e}", exc_info=True)
            return None
        
class MarkdownOutputHandler:
    """
    Obrero especializado en generar y guardar contenido Markdown,
    especialmente tablas.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        # logger.debug("MarkdownOutputHandler initialized.")

    def save_table_view(self, 
                        headers: List[str], 
                        table_matrix: List[List[Dict[str, Any]]], 
                        output_dir: str, 
                        file_name_with_extension: str,
                        document_title: Optional[str] = None) -> Optional[str]:
        """
        Formatea una tabla como Markdown y la guarda en un archivo.

        Args:
            headers: Lista de encabezados de columna.
            table_matrix: Matriz de la tabla (lista de filas, donde cada fila es lista de celdas-dict).
            output_dir: Directorio de salida.
            file_name_with_extension: Nombre del archivo Markdown.
            document_title: Título opcional para el documento Markdown.

        Returns:
            La ruta completa al archivo guardado si tiene éxito, None en caso contrario.
        """
        markdown_content_lines = []
        if document_title:
            markdown_content_lines.append(f"# Tabla Extraída para: {document_title}\n")

        markdown_table_str = TableFormatter.format_as_markdown(headers, table_matrix) #
        markdown_content_lines.append(markdown_table_str)

        full_markdown_content = "\n".join(markdown_content_lines)

        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name_with_extension)
            with open(output_path, 'w', encoding='utf-8') as f_md:
                f_md.write(full_markdown_content)
            logger.info(f"Vista de tabla Markdown guardada en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error guardando Markdown de tabla en {output_path}: {e}", exc_info=True)
            return None
#futuros módulos
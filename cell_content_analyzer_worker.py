# PerfectOCR/core/table_extractor/cell_content_analyzer_worker.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CellContentAnalyzerWorker:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("CellContentAnalyzerWorker inicializado.")

    def analyze_cell_structure(self, words_in_cell: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula métricas estadísticas (DE, Covarianza) para el contenido de una celda.
        Espera que cada palabra en words_in_cell tenga 'original_geometry_data' con 'cx' y 'cy',
        o como fallback 'vector_representation', o 'cx' y 'cy' directamente.
        """
        analysis_results = {
            "content_std_dev_x": None,
            "content_std_dev_y": None,
            "content_covariance_matrix": None
        }

        if not words_in_cell:
            return analysis_results

        # Asegurar que estas listas estén definidas antes del bucle
        word_centroids_x: List[float] = [] # Inicialización
        word_centroids_y: List[float] = [] # Inicialización

        for w_dict in words_in_cell:
            cx, cy = None, None
            original_geom = w_dict.get('original_geometry_data', {})
            
            if isinstance(original_geom, dict) and 'cx' in original_geom and 'cy' in original_geom:
                try:
                    cx = float(original_geom['cx'])
                    cy = float(original_geom['cy'])
                except (TypeError, ValueError):
                    logger.warning(f"Coordenadas de centroide no numéricas en original_geom para '{w_dict.get('text', 'N/A')}': cx={original_geom.get('cx')}, cy={original_geom.get('cy')}")
                    cx, cy = None, None # Asegurar que se resetean si la conversión falla
            
            if cx is None and cy is None and 'cx' in w_dict and 'cy' in w_dict: # Buscar directamente en w_dict si no se encontró en original_geom
                try:
                    cx = float(w_dict['cx'])
                    cy = float(w_dict['cy'])
                except (TypeError, ValueError):
                    logger.warning(f"Coordenadas de centroide no numéricas directamente en w_dict para '{w_dict.get('text', 'N/A')}': cx={w_dict.get('cx')}, cy={w_dict.get('cy')}")
                    cx, cy = None, None # Asegurar que se resetean
            
            if cx is None and cy is None: # Fallback a vector_representation
                vector_rep = w_dict.get('vector_representation')
                if vector_rep and isinstance(vector_rep, list) and len(vector_rep) >= 2:
                    try: 
                        cx_cand, cy_cand = vector_rep[0], vector_rep[1]
                        if isinstance(cx_cand, (int, float)) and isinstance(cy_cand, (int, float)):
                            cx = float(cx_cand)
                            cy = float(cy_cand)
                        else:
                            logger.warning(f"Vector representation para '{w_dict.get('text', 'N/A')}' contiene valores no numéricos: {vector_rep}")
                    except IndexError:
                        logger.warning(f"Vector representation para '{w_dict.get('text', 'N/A')}' es demasiado corto: {vector_rep}")
            
            if cx is not None and cy is not None:
                # Este try-except ya estaba, es para el .append()
                try:
                    word_centroids_x.append(float(cx))
                    word_centroids_y.append(float(cy))
                except (TypeError, ValueError): # Aunque ya deberían ser float, es una doble verificación.
                    logger.warning(f"Valor final de cx o cy no convertible a float para palabra '{w_dict.get('text', 'N/A')}' antes de append.")
            else:
                logger.debug(f"No se pudo obtener centroide para palabra '{w_dict.get('text', 'N/A')}' en análisis de celda.")

        num_valid_centroids = len(word_centroids_x)

        if num_valid_centroids > 0:
            try:
                analysis_results['content_std_dev_x'] = round(float(np.std(word_centroids_x, ddof=1 if num_valid_centroids > 1 else 0)), 3)
                analysis_results['content_std_dev_y'] = round(float(np.std(word_centroids_y, ddof=1 if num_valid_centroids > 1 else 0)), 3)
            except Exception as e_std:
                logger.error(f"Error calculando desviación estándar de centroides: {e_std}")


        if num_valid_centroids >= 2: 
            try:
                covariance_matrix = np.cov(np.array([word_centroids_x, word_centroids_y]))
                if covariance_matrix.shape == (2,2): 
                    analysis_results['content_covariance_matrix'] = [[round(float(val), 3) for val in row] for row in covariance_matrix.tolist()]
                else:
                    logger.warning(f"Matriz de covarianza con forma inesperada: {covariance_matrix.shape}")
            except Exception as e_cov:
                logger.error(f"Error calculando matriz de covarianza: {e_cov}")
        
        return analysis_results
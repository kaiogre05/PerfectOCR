# PerfectOCR/core/table_extractor/row_definer.py
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class RowDefiner:
    def __init__(self, config: Dict):
        self.config = config
        self.distance_threshold_factor = float(config.get('row_cluster_distance_threshold_factor', {})) 
        self.default_word_height = float(config.get('default_row_height_fallback', {}))
        logger.debug(f"RowDefiner inicializado con distance_threshold_factor: {self.distance_threshold_factor}, default_word_height: {self.default_word_height}")

    def _filter_data_words_and_get_avg_height(self, 
                                              all_document_words_vectorized: List[Dict[str, Any]], 
                                              y_max_header_band: float, 
                                              y_min_table_end: Optional[float]) -> Tuple[List[Dict[str, Any]], float]:
        data_words_candidates = []
        valid_heights = []

        for word_vec in all_document_words_vectorized:
            vector_rep = word_vec.get('vector_representation')
            original_geom = word_vec.get('original_geometry_data', {})

            if not vector_rep or not isinstance(vector_rep, list) or len(vector_rep) < 2:
                logger.debug(f"Palabra '{word_vec.get('text', 'N/A')}' sin 'vector_representation' válida, omitiendo de filtro de filas.")
                continue
            
            word_cy = vector_rep[1] # Coordenada 'y' del centroide del vector
            word_height = original_geom.get('height') # Altura desde datos originales

            if word_height is not None and isinstance(word_height, (int, float)) and word_height > 0:
                valid_heights.append(float(word_height))
            else:
                # No añadir a valid_heights, pero la palabra aún podría ser candidata si su 'cy' es válido.
                logger.debug(f"Palabra '{word_vec.get('text', 'N/A')}' sin 'height' válida en original_geometry_data ({word_height}).")
            
            # Filtrar por posición vertical
            if word_cy > y_max_header_band:
                if y_min_table_end is None or word_cy < y_min_table_end:
                    data_words_candidates.append(word_vec) 
        
        if not data_words_candidates:
            return [], self.default_word_height

        avg_height = np.median(valid_heights) if valid_heights else self.default_word_height
        if avg_height <= 0: 
            avg_height = self.default_word_height
            logger.warning(f"Altura promedio/mediana de palabra inválida o cero ({avg_height}), usando fallback: {self.default_word_height}")
        
        # logger.debug(f"_filter_data_words: {len(data_words_candidates)} candidatos, altura media/mediana: {avg_height:.2f}")
        return data_words_candidates, float(avg_height)

    def define_data_rows(self,
                        all_document_words: List[Dict[str, Any]], # Recibe lista de elementos vectorizados
                        y_max_header_band: float,
                        y_min_table_end: Optional[float]) -> List[Dict[str, Any]]:

        data_words_filtered_vectorized, avg_word_height = self._filter_data_words_and_get_avg_height(
            all_document_words, y_max_header_band, y_min_table_end
        )

        if not data_words_filtered_vectorized:
            logger.warning("No se encontraron palabras de datos (vectorizadas y filtradas) para definir filas.")
            return []
        
        y_centroids_for_clustering = []
        valid_data_words_for_clustering = [] # Mantener correspondencia con y_centroids_for_clustering
        for word_vec in data_words_filtered_vectorized:
            vector_rep = word_vec.get('vector_representation')
            if vector_rep and isinstance(vector_rep, list) and len(vector_rep) >=2 and isinstance(vector_rep[1], (int,float)):
                y_centroids_for_clustering.append(vector_rep[1]) 
                valid_data_words_for_clustering.append(word_vec)
            else:
                logger.warning(f"Elemento vectorizado '{word_vec.get('text', 'N/A')}' sin vector_representation[1] numérico y válido, omitiendo de clustering de filas.")
        
        if not valid_data_words_for_clustering:
            logger.warning("No hay palabras válidas con representación vectorial para clustering de filas después del filtrado final.")
            return []

        y_centroids_np = np.array(y_centroids_for_clustering).reshape(-1, 1)
        
        # Asegurar que distance_thresh sea positivo
        distance_thresh = avg_word_height * self.distance_threshold_factor
        if distance_thresh <= 0:
            logger.warning(f"Distance_threshold ({distance_thresh}) no es positivo. Usando fallback = avg_word_height ({avg_word_height}) / 2.")
            distance_thresh = max(avg_word_height / 2.0, 1.0) # Asegurar que sea al menos 1.0

        # logger.debug(f"Definiendo filas con {len(valid_data_words_for_clustering)} palabras, umbral de distancia: {distance_thresh:.2f}")

        # El clustering puede fallar si y_centroids_np está vacío o tiene una sola muestra y n_clusters=None no puede manejarlo.
        if y_centroids_np.shape[0] == 0:
            logger.warning("Array de centroides Y para clustering está vacío. No se pueden definir filas.")
            return []
        if y_centroids_np.shape[0] == 1: # Si solo hay una palabra, es una sola fila
            row_labels = np.array([0])
        else:
            agg_clustering = AgglomerativeClustering(n_clusters=None,
                                                     distance_threshold=distance_thresh,
                                                     linkage='average') 
            try:
                row_labels = agg_clustering.fit_predict(y_centroids_np)
            except ValueError as e:
                logger.error(f"Error durante el clustering aglomerativo en RowDefiner: {e}. Datos: {y_centroids_np.tolist()}")
                return []

        defined_rows_intermediate: List[Dict[str, Any]] = []
        unique_row_labels = sorted(list(set(row_labels)))

        for label in unique_row_labels:
            current_row_words_vectorized = [valid_data_words_for_clustering[i] for i, cluster_label in enumerate(row_labels) if cluster_label == label]
            if not current_row_words_vectorized:
                continue

            current_row_words_vectorized.sort(key=lambda w_vec: w_vec.get('vector_representation', [0,0])[0] 
                                             if w_vec.get('vector_representation') and isinstance(w_vec['vector_representation'], list) and len(w_vec['vector_representation']) > 0 else 0)

            row_y_centroids = []
            row_ymins = []
            row_ymaxs = []
            
            for word_vec in current_row_words_vectorized:
                vector_rep = word_vec.get('vector_representation')
                original_geom = word_vec.get('original_geometry_data', {})
                
                # Centroide Y de la fila
                if vector_rep and isinstance(vector_rep, list) and len(vector_rep) >= 2 and isinstance(vector_rep[1], (int,float)):
                    row_y_centroids.append(vector_rep[1])
                
                # ymin, ymax de la fila desde la geometría original respaldada
                ymin_w = original_geom.get('ymin')
                ymax_w = original_geom.get('ymax')

                # Asegurar que ymin_w y ymax_w sean numéricos
                if ymin_w is not None and isinstance(ymin_w, (int, float)): row_ymins.append(float(ymin_w))
                if ymax_w is not None and isinstance(ymax_w, (int, float)): row_ymaxs.append(float(ymax_w))

            if not row_y_centroids or not row_ymins or not row_ymaxs:
                text_preview = " ".join([w.get('text', 'N/A') for w in current_row_words_vectorized[:3]])
                logger.warning(f"Faltan datos geométricos para calcular propiedades de fila (label {label}, texto: '{text_preview}...'). Omitiendo fila.")
                continue

            Y_r = np.mean(row_y_centroids)
            ymin_row = min(row_ymins)
            ymax_row = max(row_ymaxs)
            DeltaY_r = ymax_row - ymin_row
            
            max_preview_words = 15 
            row_text_for_log = " ".join([word_vec.get('text', '') for word_vec in current_row_words_vectorized[:max_preview_words]])
            if len(current_row_words_vectorized) > max_preview_words:
                row_text_for_log += "..."

            defined_rows_intermediate.append({
                "Y_r": Y_r,
                "DeltaY_r": DeltaY_r,
                "ymin_row": ymin_row,
                "ymax_row": ymax_row,
                "words_in_row": current_row_words_vectorized, # Guardar los elementos vectorizados
                "_text_for_log": row_text_for_log 
            })

        defined_rows_intermediate.sort(key=lambda r: r['Y_r'])

        final_defined_rows: List[Dict[str, Any]] = []
        for i, row_data in enumerate(defined_rows_intermediate):
            row_data['row_index'] = i
            logger.info(f"Fila {i:03d} - {row_data['_text_for_log']}") # Log original
            if '_text_for_log' in row_data:
                del row_data['_text_for_log'] 
            final_defined_rows.append(row_data) 

        logger.info(f"Definidas {len(final_defined_rows)} filas de datos en total (procesando elementos vectorizados).")
        return final_defined_rows
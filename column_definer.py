import logging
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict

logger = logging.getLogger(__name__)

class ColumnDefiner:
    def __init__(self, config: Dict):
        self.config = config
        self.dbscan_eps = float(config.get('dbscan_header_eps', {})) 
        self.dbscan_min_samples = int(config.get('dbscan_header_min_samples', {}))

    def define_table_columns(self, header_words: List[Dict]) -> List[Dict]:
        #logger.info(f"Definiendo columnas de tabla a partir de {len(header_words)} palabras de encabezado.")
        if not header_words:
            return []

        # Asegurar que las palabras tengan 'cx' (centroide x)
        # (esto debería venir de _get_words_from_lines en HeaderDetector o de TextElement)
        header_words_with_cx = [w for w in header_words if 'cx' in w]
        if not header_words_with_cx:
            logger.warning("Palabras de encabezado no tienen centroides 'cx'.")
            return []
            
        x_centroids = np.array([word['cx'] for word in header_words_with_cx]).reshape(-1, 1)

        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(x_centroids)

        defined_columns = []
        unique_labels = sorted(list(set(cluster_labels)))

        for label in unique_labels:
            if label == -1:  # Omitir ruido si DBSCAN lo produce
                # Podrías tener una lógica para manejar palabras ruidosas (asignarlas a la columna más cercana si están aisladas)
                continue

            column_member_words = [header_words_with_cx[i] for i, lbl in enumerate(cluster_labels) if lbl == label]
            if not column_member_words:
                continue

            # Ordenar palabras dentro de la columna (útil para encabezados multilínea y para texto)
            column_member_words.sort(key=lambda w: (w.get('cy', 0), w['cx']))

            # Calcular propiedades de la columna
            X_k = np.mean([word['cx'] for word in column_member_words])
            Y_H_k = np.mean([word['cy'] for word in column_member_words]) # Centroide Y del texto del encabezado
            
            xmin_col = min(word.get('xmin', word['cx']) for word in column_member_words)
            xmax_col = max(word.get('xmax', word['cx']) for word in column_member_words)
            DeltaX_k = xmax_col - xmin_col
            
            H_k_text = " ".join([word['text'] for word in column_member_words])

            defined_columns.append({
                "H_k_text": H_k_text,
                "X_k": X_k,
                "DeltaX_k": DeltaX_k,
                "Y_H_k": Y_H_k,
                "xmin_col": xmin_col,
                "xmax_col": xmax_col,
                "constituent_table_keywords_list": column_member_words # Para depuración o lógica avanzada
            })

        # Ordenar columnas por su posición X
        defined_columns.sort(key=lambda c: c['X_k'])
        for i, col in enumerate(defined_columns): col['column_index'] = i
            
        logger.info(f"Definidas {len(defined_columns)} columnas: {[c['H_k_text'] for c in defined_columns]}")
        return defined_columns
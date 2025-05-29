# PerfectOCR/core/input_validation/quality_evaluator.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image as PILImage
from core.preprocessing.transformers import high_precision_skew_correction

logger = logging.getLogger(__name__)

class ImageQualityEvaluator:
    def __init__(self, config: Dict):
        self.config = config 
        qa_params_cfg = self.config.get('quality_assessment_params', {})
        deskew_cfg = self.config.get('deskew', {})

        self.min_dpi_log_threshold = qa_params_cfg.get('min_dpi_threshold', {})
        self.sharpness_log_threshold = qa_params_cfg.get('sharpness_threshold_for_assessment', {})
        self.contrast_log_threshold = qa_params_cfg.get('contrast_std_dev_threshold_for_assessment', {})
        self.skew_detection_inform_threshold = qa_params_cfg.get('skew_detection_inform_threshold', {})
        self.severe_skew_log_threshold = qa_params_cfg.get('severe_skew_degrees_for_assessment', {})

        self.canny_thresh1_detect = deskew_cfg.get('canny_threshold1', {})
        self.canny_thresh2_detect = deskew_cfg.get('canny_threshold2', {})
        self.canny_aperture_detect = deskew_cfg.get('canny_aperture_size', {})
        self.hough_min_line_cap_detect = deskew_cfg.get('hough_min_line_length_cap_px', {})
        self.hough_max_gap_detect = deskew_cfg.get('hough_max_line_gap_px', {})
        self.hough_thresh_detect = deskew_cfg.get('hough_threshold', {})
        self.hough_angle_range_detect = deskew_cfg.get('hough_angle_filter_range_degrees', [-20, 20])
        hough_angle_range_default = deskew_cfg.get('hough_angle_filter_range_degrees', [-20.0, 20.0])
        if isinstance(hough_angle_range_default, list) and len(hough_angle_range_default) == 2:
            self.hough_angle_range_detect = [float(hough_angle_range_default[0]), float(hough_angle_range_default[1])]
        else:
            self.hough_angle_range_detect = [-20.0, 20.0]   


        logger.info(f"ImageQualityEvaluator (para métricas) inicializado. "
                    f"Umbrales informativos de log: DPI<{self.min_dpi_log_threshold}, "
                    f"Nitidez<{self.sharpness_log_threshold}, Contraste<{self.contrast_log_threshold}, "
                    f"InclinaciónSevera>{self.severe_skew_log_threshold}°")
        
    def evaluate_image_metrics(self, image_array: np.ndarray, input_path_for_log: str) -> Tuple[Dict[str, Any], List[str]]:
        quality_assessment_metrics: Dict[str, Any] = {'input_path_reference': input_path_for_log}
        quality_observations: List[str] = []

        if image_array is None or image_array.size == 0:
            quality_assessment_metrics['error'] = "Image array is empty or None"
            quality_observations.append("error_loading_image_array")
            return quality_assessment_metrics, quality_observations

        try:
            height, width = image_array.shape[:2]
            num_channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
            quality_assessment_metrics['dimensions'] = {'height': height, 'width': width, 'channels': num_channels}

            try:
                pil_img_from_array = PILImage.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if num_channels == 3 else image_array)
                quality_assessment_metrics['pil_mode_from_array'] = pil_img_from_array.mode
                quality_assessment_metrics['dpi'] = "unknown_from_array"
                if pil_img_from_array.mode == 'P':
                    palette = pil_img_from_array.getpalette()
                    if palette:
                        is_gray_palette = not any(palette[i] != palette[i+1] or palette[i] != palette[i+2] for i in range(0, min(len(palette), 768), 3))
                        quality_assessment_metrics['pil_palette_is_grayscale'] = is_gray_palette
                        if not is_gray_palette: quality_observations.append("color_palette_detected")
            except Exception as e_pil_array:
                logger.debug(f"No se pudieron obtener metadatos PIL del array de imagen: {e_pil_array}")
                quality_assessment_metrics['pil_mode_from_array'] = "error"

            gray_for_analysis_internals: Optional[np.ndarray] = None
            if num_channels == 4:
                gray_for_analysis_internals = cv2.cvtColor(image_array, cv2.COLOR_BGRA2GRAY)
            elif num_channels == 3:
                gray_for_analysis_internals = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            elif num_channels == 1:
                gray_for_analysis_internals = image_array.copy()
            else:
                quality_assessment_metrics['error_color_conversion'] = f"Unsupported number of channels: {num_channels}"
                quality_observations.append("unsupported_image_channels")
                return quality_assessment_metrics, quality_observations

            if gray_for_analysis_internals is not None:
                # Calcular nitidez y contraste
                laplacian_var = cv2.Laplacian(gray_for_analysis_internals, cv2.CV_64F).var()
                quality_assessment_metrics['sharpness_laplacian_variance'] = round(laplacian_var, 2)
                if laplacian_var < self.sharpness_log_threshold:
                    quality_observations.append(f"low_sharpness_observed (LapVar: {laplacian_var:.2f})")

                std_intensity = np.std(gray_for_analysis_internals)
                quality_assessment_metrics['contrast_std_dev'] = round(std_intensity, 2)
                if std_intensity < self.contrast_log_threshold:
                    quality_observations.append(f"low_contrast_observed (StdDev: {std_intensity:.2f})")

                # Calcular inclinación (una sola vez y correctamente)
                try:
                    _, detected_angle = high_precision_skew_correction(
                        gray_img=gray_for_analysis_internals.copy(),
                        skew_threshold=self.skew_detection_inform_threshold,
                        canny_thresh1=self.canny_thresh1_detect,
                        canny_thresh2=self.canny_thresh2_detect,
                        canny_aperture=self.canny_aperture_detect,
                        hough_min_line_len_cap=self.hough_min_line_cap_detect,
                        hough_max_gap=self.hough_max_gap_detect,
                        hough_thresh=self.hough_thresh_detect,
                        hough_angle_range=self.hough_angle_range_detect
                    )
                    quality_assessment_metrics['estimated_skew_angle_degrees'] = round(detected_angle, 2)

                    if abs(detected_angle) > self.severe_skew_log_threshold:
                        quality_observations.append(f"severe_skew_angle_observed ({detected_angle:.2f}°)")
                    elif abs(detected_angle) > self.skew_detection_inform_threshold:
                        quality_observations.append(f"skew_angle_observed ({detected_angle:.2f}°)")

                except TypeError as te:
                    logger.error(f"TypeError al llamar a high_precision_skew_correction: {te}.")
                    quality_assessment_metrics['estimated_skew_angle_degrees'] = 0.0
                    quality_observations.append("error_skew_detection_type_error")
                except Exception as e_skew:
                    logger.error(f"Excepción en detección de inclinación: {e_skew}", exc_info=True)
                    quality_assessment_metrics['estimated_skew_angle_degrees'] = 0.0
                    quality_observations.append("error_in_skew_detection")
            # Fin del if gray_for_analysis_internals is not None:
        except Exception as e: # Este es el except del try principal
            logger.error(f"Excepción en ImageQualityEvaluator.evaluate_image_metrics para {input_path_for_log}: {e}", exc_info=True)
            quality_assessment_metrics['error'] = f"Exception during quality metric evaluation: {str(e)}"
            quality_observations.append("error_in_quality_evaluation")
            return quality_assessment_metrics, quality_observations
        return quality_assessment_metrics, sorted(list(set(quality_observations)))

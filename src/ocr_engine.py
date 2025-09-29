import cv2
import pytesseract
import numpy as np
import os
import time
from src.config import OCR_LANG, OCR_CONFIDENCE_THRESHOLD, OCR_TIMEOUT
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self):
        self.lang = OCR_LANG
        self.confidence_threshold = OCR_CONFIDENCE_THRESHOLD
        self.timeout = OCR_TIMEOUT

        # Verify tesseract installation
        self.tesseract_available = False
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
            self.tesseract_available = True
        except Exception as e:
            logger.warning(f"Tesseract not properly installed: {e}")
            logger.warning("OCR functionality will be limited until Tesseract is installed")

    def extract_text(self, image_path: str) -> tuple[str, int, int]:
        """
        Extract text from image with preprocessing.
        Returns: (extracted_text, confidence_score, processing_time_ms)
        """
        start_time = time.time()

        if not self.tesseract_available:
            logger.warning("Tesseract not available, cannot extract text")
            return "", 0, int((time.time() - start_time) * 1000)

        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return "", 0, 0

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return "", 0, 0

            # Apply preprocessing
            processed_image = self._preprocess_image(image)

            # Extract text with detailed data
            custom_config = f'--oem 3 --psm 6 -l {self.lang}'
            data = pytesseract.image_to_data(
                processed_image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            # Filter and combine text based on confidence
            text_parts = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if int(conf) > self.confidence_threshold:
                    word = data['text'][i].strip()
                    if word and len(word) > 1:
                        text_parts.append(word)
                        confidences.append(int(conf))

            extracted_text = ' '.join(text_parts)
            avg_confidence = int(np.mean(confidences)) if confidences else 0
            processing_time = int((time.time() - start_time) * 1000)

            logger.debug(f"OCR extracted {len(text_parts)} words, avg confidence: {avg_confidence}%")
            return extracted_text, avg_confidence, processing_time

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OCR failed for {image_path}: {e}")
            return "", 0, processing_time

    def _preprocess_image(self, image):
        """Apply preprocessing pipeline for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding for better text separation
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Additional noise reduction with opening operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        return opening

    def health_check(self) -> bool:
        """Verify OCR engine is working correctly"""
        try:
            # Create simple test image
            test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, 'TEST', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Test OCR
            result = pytesseract.image_to_string(test_image, lang=self.lang)
            return 'TEST' in result.upper()

        except Exception as e:
            logger.error(f"OCR health check failed: {e}")
            return False

    def get_supported_languages(self) -> list:
        """Get list of supported OCR languages"""
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return []

    def extract_text_from_bytes(self, image_bytes: bytes) -> tuple[str, int, int]:
        """
        Extract text from image bytes (useful for processing downloaded images).
        Returns: (extracted_text, confidence_score, processing_time_ms)
        """
        start_time = time.time()

        if not self.tesseract_available:
            logger.warning("Tesseract not available, cannot extract text from bytes")
            return "", 0, int((time.time() - start_time) * 1000)

        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error("Failed to decode image from bytes")
                return "", 0, 0

            # Apply preprocessing
            processed_image = self._preprocess_image(image)

            # Extract text with detailed data
            custom_config = f'--oem 3 --psm 6 -l {self.lang}'
            data = pytesseract.image_to_data(
                processed_image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            # Filter and combine text based on confidence
            text_parts = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if int(conf) > self.confidence_threshold:
                    word = data['text'][i].strip()
                    if word and len(word) > 1:
                        text_parts.append(word)
                        confidences.append(int(conf))

            extracted_text = ' '.join(text_parts)
            avg_confidence = int(np.mean(confidences)) if confidences else 0
            processing_time = int((time.time() - start_time) * 1000)

            logger.debug(f"OCR extracted {len(text_parts)} words from bytes, avg confidence: {avg_confidence}%")
            return extracted_text, avg_confidence, processing_time

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OCR failed for image bytes: {e}")
            return "", 0, processing_time
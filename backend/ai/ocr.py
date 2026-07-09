"""OCR pipeline: PDF/image → raw text extraction.

Uses pytesseract as the primary OCR engine (reliable, easy Docker install).
Falls back to LLM-based extraction via Fireworks API for better accuracy.
"""

import os
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images."""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=300)
        logger.info(f"Converted PDF to {len(images)} page(s)")
        return images
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise ValueError(f"Failed to convert PDF: {e}")


def image_to_text_tesseract(image: Image.Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a PDF or image file.
    
    Returns the concatenated raw text from all pages.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        images = pdf_to_images(file_path)
        texts = []
        for i, img in enumerate(images):
            logger.info(f"Processing page {i + 1}/{len(images)}")
            text = image_to_text_tesseract(img)
            if text:
                texts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(texts)
    
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        img = Image.open(file_path)
        return image_to_text_tesseract(img)
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def is_ocr_available() -> bool:
    """Check if Tesseract OCR is available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

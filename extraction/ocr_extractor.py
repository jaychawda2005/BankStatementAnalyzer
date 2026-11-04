import os
import shutil
import io
import warnings
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_WARNED_CV2 = False


def _find_tesseract_binary():
    """Locate tesseract binary on the system or return None."""
    # 1. Check if already in PATH
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
        
    # 2. Check common Windows installation locations
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
            
    return None


def is_ocr_available():
    """Return True if Tesseract OCR is installed and accessible."""
    return _find_tesseract_binary() is not None


def deskew_image(image, max_angle=5.0, step=0.25):
    """
    Detect and correct orientation skew in scanned documents.
    Uses horizontal projection profile variance optimization.
    When rows of text are horizontal, row-sum variance is maximized.
    """
    global _WARNED_CV2
    if not _HAS_CV2:
        if not _WARNED_CV2:
            warnings.warn(
                "OpenCV (cv2) is not installed; automatic deskewing is disabled.",
                RuntimeWarning,
                stacklevel=2
            )
            _WARNED_CV2 = True
        return image, 0.0

    # Convert PIL Image to grayscale numpy array
    img_gray = np.array(image.convert('L'))
    h, w = img_gray.shape

    # Scale down for fast angle search if image is very large
    scale = min(1.0, 900.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_gray

    # Otsu thresholding: invert so text is 255 and background is 0
    _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    sh, sw = thresh.shape
    center = (sw // 2, sh // 2)

    best_angle = 0.0
    max_variance = -1.0

    # Search angles in [-max_angle, max_angle]
    angles = np.arange(-max_angle, max_angle + step, step)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            thresh, M, (sw, sh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        proj = np.sum(rotated, axis=1, dtype=np.float64)
        var = np.var(proj)
        if var > max_variance:
            max_variance = var
            best_angle = float(angle)

    # If significant skew detected (> 0.1 degree), rotate original image
    if abs(best_angle) > 0.1:
        orig_center = (w // 2, h // 2)
        M_full = cv2.getRotationMatrix2D(orig_center, best_angle, 1.0)
        deskewed = cv2.warpAffine(
            img_gray, M_full, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )
        return Image.fromarray(deskewed), best_angle

    return image, 0.0


def preprocess_image_for_ocr(image):
    """
    Full OCR preprocessing pipeline:
    1. Automatic deskewing via projection profile variance
    2. Grayscale conversion
    3. Contrast enhancement & adaptive binarization
    4. Mild sharpening to clarify degraded numbers and dashes
    """
    # 1. Automatic Deskew
    deskewed_img, _ = deskew_image(image)

    # 2. Grayscale
    gray = deskewed_img.convert('L')

    # 3. Contrast enhancement
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.8)

    # 4. Sharpening
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return sharpened


def convert_pdf_to_images(file_input, dpi=300):
    """
    Convert PDF pages to PIL Images using PyMuPDF (fitz).
    Eliminates external dependency on Poppler/pdf2image.
    Accepts file path, bytes, or file-like objects (e.g. UploadedFile).
    """
    if hasattr(file_input, 'read'):
        if hasattr(file_input, 'seek'):
            file_input.seek(0)
        data = file_input.read()
    elif isinstance(file_input, (bytes, bytearray)):
        data = file_input
    elif isinstance(file_input, str) and os.path.isfile(file_input):
        with open(file_input, 'rb') as f:
            data = f.read()
    else:
        raise ValueError("Invalid file input for PDF conversion.")

    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    
    # Render at requested DPI (default 300 DPI for crisp OCR)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
        
    doc.close()
    return images


def extract_text_via_ocr(file_input, psm=6):
    """
    Extract text from a scanned/image PDF via OCR.
    Uses PyMuPDF for page rendering (no Poppler needed) and Pytesseract for OCR.
    
    Raises RuntimeError with clear guidance if Tesseract is not installed.
    """
    tess_path = _find_tesseract_binary()
    if not tess_path:
        raise RuntimeError(
            "Tesseract OCR is not installed or not found on PATH. "
            "Please install Tesseract OCR (Windows: UB-Mannheim installer) "
            "to process scanned image-based bank statements."
        )

    images = convert_pdf_to_images(file_input, dpi=300)
    extracted_text = []

    # PSM 6 = Assume a single uniform block of text (ideal for tabular bank statements)
    custom_config = f'--psm {psm}'

    for i, img in enumerate(images):
        processed = preprocess_image_for_ocr(img)
        page_text = pytesseract.image_to_string(processed, config=custom_config)
        extracted_text.append(page_text)

    return "\n".join(extracted_text)


def extract_single_image_ocr(image, psm=6):
    """
    Extract OCR text from a single PIL Image.
    Used by per-page routing for scanned pages.
    """
    tess_path = _find_tesseract_binary()
    if not tess_path:
        raise RuntimeError(
            "Tesseract OCR is not installed or not found on PATH. "
            "Please install Tesseract OCR (Windows: UB-Mannheim installer) "
            "to process scanned image-based bank statements."
        )
    processed = preprocess_image_for_ocr(image)
    return pytesseract.image_to_string(processed, config=f'--psm {psm}')


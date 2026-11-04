import fitz  # PyMuPDF

def inspect_pdf_pages(file_bytes):
    """
    Performs per-page inspection of a PDF document.
    Categorizes each individual page as 'text', 'scanned', or 'empty'.
    
    Returns:
        dict: {
            "overall_type": "text" | "scanned" | "mixed",
            "total_pages": int,
            "page_details": list of dicts with page-level metrics
        }
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes). Please upload a valid bank statement PDF.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except fitz.FileDataError:
        raise ValueError("The uploaded file is corrupted or not a valid PDF document.")
    except Exception as e:
        raise ValueError(f"Failed to open PDF document: {str(e)}")

    try:
        if doc.is_encrypted or doc.needs_pass:
            raise ValueError("This PDF is password-protected. Please remove password protection before uploading.")

        if len(doc) == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        page_details = []
        text_pages = 0
        scanned_pages = 0

        for idx, page in enumerate(doc):
            text = page.get_text().strip()
            images = len(page.get_images())
            char_count = len(text)

            # A page is 'text' if it has readable digital text characters (>= 50 chars)
            # A page is 'scanned' if it has minimal text (< 50 chars) but has images
            if char_count >= 50:
                p_type = "text"
                text_pages += 1
            elif images > 0:
                p_type = "scanned"
                scanned_pages += 1
            else:
                p_type = "empty"

            page_details.append({
                "page": idx + 1,
                "type": p_type,
                "char_count": char_count,
                "image_count": images
            })

        # Overall classification
        if text_pages > 0 and scanned_pages > 0:
            overall_type = "mixed"
        elif scanned_pages > 0 and text_pages == 0:
            overall_type = "scanned"
        elif text_pages > 0:
            overall_type = "text"
        else:
            raise ValueError("The uploaded PDF is empty. No text, tables, or document images were found.")

        return {
            "overall_type": overall_type,
            "total_pages": len(doc),
            "text_pages": text_pages,
            "scanned_pages": scanned_pages,
            "page_details": page_details
        }

    finally:
        doc.close()


def detect_pdf_type(file_bytes):
    """
    Convenience function returning 'text', 'scanned', or 'mixed'.
    Maintains backward compatibility with the existing API.
    """
    info = inspect_pdf_pages(file_bytes)
    return info["overall_type"]

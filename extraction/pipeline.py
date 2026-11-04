import io
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image

from extraction.pdf_detector import inspect_pdf_pages
from extraction.text_extractor import (
    extract_text_from_pdf,
    extract_tables_from_pdf,
    extract_transactions_from_page
)
from extraction.ocr_extractor import (
    is_ocr_available,
    extract_text_via_ocr,
    extract_single_image_ocr
)
from extraction.transaction_parser import (
    parse_account_details,
    parse_transactions,
    parse_ocr_transactions
)


def extract_document(file_bytes, ocr_handler=None):
    """
    Unified, intelligent document extraction pipeline.
    
    Performs per-page routing:
    - Text pages are extracted via structured table parsers (pdfplumber) or regex.
    - Scanned/image pages are rendered to high-resolution bitmaps and processed via OCR.
    - If Tesseract is not installed and scanned pages exist, text pages are preserved
      and a clear warning is returned identifying the skipped pages.
    - Transactions and narratives are merged in exact chronological/page order.
    - Account metadata (Account Name, Number, IFSC, Bank) is extracted from combined text.
    
    Parameters:
        file_bytes (bytes or file-like): PDF file contents.
        ocr_handler (callable, optional): Custom OCR callable for testing or alternative OCR engines.
                                          Signature: ocr_handler(image, page_num) -> str.
                                          
    Returns:
        tuple: (raw_df: pd.DataFrame, raw_text: str, account_details: dict, method_description: str)
    """
    # Normalize input to bytes
    if hasattr(file_bytes, 'read'):
        if hasattr(file_bytes, 'seek'):
            file_bytes.seek(0)
        data = file_bytes.read()
    elif isinstance(file_bytes, (bytes, bytearray)):
        data = bytes(file_bytes)
    else:
        raise ValueError("Invalid file input: expected bytes or file-like object.")

    # 1. Page-level inspection & document classification
    pdf_info = inspect_pdf_pages(data)
    overall_type = pdf_info["overall_type"]
    page_details = pdf_info["page_details"]

    # 2. Case: Pure Text Document
    if overall_type == "text":
        with io.BytesIO(data) as bio:
            table_records = extract_tables_from_pdf(bio)
            
        with io.BytesIO(data) as bio:
            raw_text = extract_text_from_pdf(bio)

        account_details = parse_account_details(raw_text)

        if table_records:
            raw_df = pd.DataFrame(table_records)
            method = "Structured Table Extractor"
        elif raw_text:
            raw_df = parse_transactions(raw_text)
            method = "Text Regex Parser"
        else:
            raw_df = pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
            method = "No readable text or tables found"

        return raw_df, raw_text, account_details, method

    # 3. Case: Pure Scanned Document
    can_ocr = is_ocr_available() or (ocr_handler is not None)
    if overall_type == "scanned":
        if not can_ocr:
            empty_df = pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
            method = "Warning: Tesseract OCR is not installed or not in PATH. Scanned image PDF could not be processed."
            account_details = {
                "Account Name": "Not Found",
                "Account Number": "Not Found",
                "IFSC Code": "Not Found",
                "Bank Name": "Unknown Bank"
            }
            return empty_df, "", account_details, method

        if ocr_handler is not None:
            # Render images and apply handler
            fitz_doc = fitz.open(stream=data, filetype="pdf")
            zoom = 300.0 / 72.0
            mat = fitz.Matrix(zoom, zoom)
            ocr_parts = []
            for idx, page in enumerate(fitz_doc):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_parts.append(ocr_handler(img, idx + 1))
            fitz_doc.close()
            raw_text = "\n".join(ocr_parts)
        else:
            raw_text = extract_text_via_ocr(data)

        account_details = parse_account_details(raw_text)
        raw_df = parse_ocr_transactions(raw_text)
        method = "Tesseract OCR (Deskewed) + Resilient Line Parser"
        return raw_df, raw_text, account_details, method

    # 4. Case: Mixed Document (Per-page routing)
    all_transactions = []
    text_chunks = []
    text_pages_processed = []
    scanned_pages_processed = []
    skipped_scanned_pages = []

    fitz_doc = fitz.open(stream=data, filetype="pdf")
    zoom = 300.0 / 72.0
    mat = fitz.Matrix(zoom, zoom)

    with pdfplumber.open(io.BytesIO(data)) as plumb_doc:
        for idx, p_info in enumerate(page_details):
            p_num = idx + 1
            p_type = p_info["type"]

            if p_type == "text":
                # Digital text extraction
                plumb_page = plumb_doc.pages[idx]
                p_text = plumb_page.extract_text() or fitz_doc[idx].get_text() or ""
                if p_text:
                    text_chunks.append(p_text)

                page_txns = extract_transactions_from_page(plumb_page, p_text)
                if page_txns:
                    all_transactions.extend(page_txns)
                text_pages_processed.append(p_num)

            elif p_type == "scanned":
                if can_ocr:
                    # High-res image rendering
                    pix = fitz_doc[idx].get_pixmap(matrix=mat, alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    
                    if ocr_handler is not None:
                        page_ocr_text = ocr_handler(img, p_num)
                    else:
                        page_ocr_text = extract_single_image_ocr(img)

                    if page_ocr_text:
                        text_chunks.append(page_ocr_text)
                        df_page_txns = parse_ocr_transactions(page_ocr_text)
                        if not df_page_txns.empty:
                            all_transactions.extend(df_page_txns.to_dict(orient="records"))
                    scanned_pages_processed.append(p_num)
                else:
                    skipped_scanned_pages.append(p_num)

    fitz_doc.close()

    raw_text = "\n".join(text_chunks)
    account_details = parse_account_details(raw_text)

    if all_transactions:
        raw_df = pd.DataFrame(all_transactions)
    else:
        raw_df = pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])

    # Method description with clear warning if scanned pages were skipped
    if skipped_scanned_pages:
        skipped_str = ", ".join(map(str, skipped_scanned_pages))
        text_str = ", ".join(map(str, text_pages_processed))
        method = (
            f"Hybrid Routing: Extracted digital text from page(s) {text_str} "
            f"(Warning: Tesseract OCR is not installed; skipped scanned page(s): {skipped_str})"
        )
    else:
        method = (
            f"Hybrid Routing: Extracted digital text from page(s) {text_pages_processed} "
            f"and OCR from scanned page(s) {scanned_pages_processed}"
        )

    return raw_df, raw_text, account_details, method


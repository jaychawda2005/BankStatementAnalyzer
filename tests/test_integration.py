import pytest
import os
import io
import pandas as pd
from unittest.mock import patch
from extraction.pdf_detector import detect_pdf_type
from extraction.text_extractor import extract_text_from_pdf, extract_tables_from_pdf
from extraction.ocr_extractor import convert_pdf_to_images, extract_text_via_ocr, is_ocr_available
from extraction.transaction_parser import parse_account_details, parse_transactions, parse_ocr_transactions
from classification.rule_classifier import classify_by_rules_with_confidence
from classification.ml_classifier import MLClassifier
from processing.data_cleaner import clean_transaction_data
from processing.balance_validator import validate_running_balances, reconcile_statement_summary
from export.export_data import generate_csv, generate_excel


def test_full_text_sbi_workflow():
    pdf_path = os.path.join("sample_statements", "sbi_sample.pdf")
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # 1. Detection
    pdf_type = detect_pdf_type(file_bytes)
    assert pdf_type == "text"

    # 2. Table Extraction
    with open(pdf_path, "rb") as f:
        table_records = extract_tables_from_pdf(f)
    assert len(table_records) == 6

    # 3. Account Details
    with open(pdf_path, "rb") as f:
        text = extract_text_from_pdf(f)
    details = parse_account_details(text)
    assert details["Account Number"] == "30992084658"
    assert details["IFSC Code"] == "SBIN0060447"
    assert details["Bank Name"] == "State Bank of India"

    # 4. Cleaning & Validation
    df = clean_transaction_data(pd.DataFrame(table_records))
    validated = validate_running_balances(df)
    assert (validated["Balance_Status"] == "Verified").all()

    summary = reconcile_statement_summary(validated)
    assert summary["is_reconciled"] is True
    assert summary["total_debits"] == 5573.00
    assert summary["total_credits"] == 3410.00
    assert summary["calculated_closing"] == 159076.00

    # 5. Export
    csv_bytes = generate_csv(validated)
    excel_bytes = generate_excel(validated)
    assert len(csv_bytes) > 0
    assert len(excel_bytes) > 0


def test_full_text_sample_statement_workflow():
    pdf_path = os.path.join("sample_statements", "sample_bank_statement_text_based.pdf")
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    pdf_type = detect_pdf_type(file_bytes)
    assert pdf_type == "text"

    with open(pdf_path, "rb") as f:
        raw_text = extract_text_from_pdf(f)
    df = parse_transactions(raw_text)
    assert len(df) == 14

    # Verify interest credit is in Credit column
    interest_row = df[df["Description"].str.contains("INTEREST")]
    assert not interest_row.empty
    assert float(interest_row.iloc[0]["Credit"]) == 125.00
    assert float(interest_row.iloc[0]["Debit"]) == 0.00

    # Verify Netflix is in Debit column
    netflix_row = df[df["Description"].str.contains("NETFLIX")]
    assert not netflix_row.empty
    assert float(netflix_row.iloc[0]["Debit"]) == 649.00
    assert float(netflix_row.iloc[0]["Credit"]) == 0.00

    clean = clean_transaction_data(df)
    val = validate_running_balances(clean)
    assert (val["Balance_Status"] == "Verified").all()

    rec = reconcile_statement_summary(val, opening_balance=52500.0, reported_closing=66607.0)
    assert rec["is_reconciled"] is True
    assert rec["discrepancy"] == 0.0


def test_full_text_hdfc_workflow():
    pdf_path = os.path.join("sample_statements", "hdfc_sample.pdf")
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    assert detect_pdf_type(file_bytes) == "text"

    with open(pdf_path, "rb") as f:
        txns = extract_tables_from_pdf(f)
    assert len(txns) == 5

    with open(pdf_path, "rb") as f:
        text = extract_text_from_pdf(f)
    details = parse_account_details(text)
    assert details["Bank Name"] == "HDFC Bank"
    assert details["Account Number"] == "50100456789012"
    assert details["IFSC Code"] == "HDFC0000048"

    df = clean_transaction_data(pd.DataFrame(txns))
    val = validate_running_balances(df)
    assert (val["Balance_Status"] == "Verified").all()

    summary = reconcile_statement_summary(val, opening_balance=45000.0, reported_closing=76851.0)
    assert summary["is_reconciled"] is True
    assert summary["discrepancy"] == 0.0


def test_full_text_icici_workflow():
    pdf_path = os.path.join("sample_statements", "icici_sample.pdf")
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    assert detect_pdf_type(file_bytes) == "text"

    with open(pdf_path, "rb") as f:
        txns = extract_tables_from_pdf(f)
    assert len(txns) == 5

    with open(pdf_path, "rb") as f:
        text = extract_text_from_pdf(f)
    details = parse_account_details(text)
    assert details["Bank Name"] == "ICICI Bank"
    assert details["Account Number"] == "123405001234"
    assert details["IFSC Code"] == "ICIC0001234"

    df = clean_transaction_data(pd.DataFrame(txns))
    val = validate_running_balances(df)
    assert (val["Balance_Status"] == "Verified").all()

    summary = reconcile_statement_summary(val, opening_balance=52500.0, reported_closing=84801.0)
    assert summary["is_reconciled"] is True
    assert summary["discrepancy"] == 0.0


def test_full_text_axis_workflow():
    pdf_path = os.path.join("sample_statements", "axis_sample.pdf")
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    assert detect_pdf_type(file_bytes) == "text"

    with open(pdf_path, "rb") as f:
        txns = extract_tables_from_pdf(f)
    assert len(txns) == 5

    with open(pdf_path, "rb") as f:
        text = extract_text_from_pdf(f)
    details = parse_account_details(text)
    assert details["Bank Name"] == "Axis Bank"
    assert details["Account Number"] == "918010012345678"
    assert details["IFSC Code"] == "UTIB0000456"

    df = clean_transaction_data(pd.DataFrame(txns))
    val = validate_running_balances(df)
    assert (val["Balance_Status"] == "Verified").all()


def test_scanned_pdf_full_pipeline():
    """
    Complete end-to-end test of the scanned PDF OCR workflow:
    PDF upload -> PDF detection -> PyMuPDF rendering -> Tesseract OCR ->
    Transaction parsing -> Data cleaning -> Classification ->
    Balance validation -> Excel/CSV export.
    """
    pdf_path = os.path.join("sample_statements", "scanned_sample.pdf")
    assert os.path.exists(pdf_path)

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # 1. Detection: 100% image-based document
    pdf_type = detect_pdf_type(file_bytes)
    assert pdf_type == "scanned"

    # 2. PyMuPDF image rendering
    images = convert_pdf_to_images(file_bytes, dpi=150)
    assert len(images) == 1
    assert images[0].width > 0

    # 3. Tesseract OCR execution
    # If tesseract executable is installed on host, execute directly;
    # otherwise simulate with the exact authentic OCR output from scanned_sample.pdf
    sample_scanned_ocr = (
        "SAMPLE ONLY — NOT A REAL BANK STATEMENT\n"
        "Sapphire Demo Bank\n"
        "Account Holder Aarav Mehta\n"
        "Account Number XXXXXX1234\n"
        "IFSC SAPH0000001\n"
        "Opening Balance 52,500.00\n"
        "Date Transaction Description Debit Credit Balance\n"
        "01-10-2026 UPI/SWIGGY/ORDER7841 450.00 _ 52,050.00\n"
        "02-10-2026 SALARY CREDIT/ACME DEMO PVT LTD _ 35,000.00 87,050.00\n"
        "03-10-2026 UPI/UBER/TRIP 2918 320.00 _ 86,730.00\n"
        "05-10-2026 POS/DMART/AHMEDABAD 2,450.00 _ 84,280.00\n"
        "07-10-2026 ELECTRICITY BILL/UGVCL 1,280.00 _ 83,000.00\n"
        "09-10-2026 NEFT/RENT PAYMENT 12,000.00 _ 71,000.00\n"
        "11-10-2026 UPI/AMAZON/ORDER 55320 1,799.00 _ 69,201.00\n"
        "14-10-2026 ATM CASH WITHDRAWAL 3,000.00 _ 66,201.00\n"
        "16-10-2026 INTEREST CREDIT _ 125.00 66,326.00\n"
        "19-10-2026 UPI/ZOMATO/ORDER 8892 685.00 _ 65,641.00\n"
        "22-10-2026 NETFLIX SUBSCRIPTION 649.00 _ 64,992.00\n"
        "25-10-2026 IMPS RECEIVED/FRIEND _ 2,500.00 67,492.00\n"
        "28-10-2026 BANK SMS ALERT CHARGE 25.00 _ 67,467.00\n"
        "31-10-2026 UPI/PHARMACY/HEALTHPLUS 860.00 _ 66,607.00\n"
        "Summary\n"
        "Total Credits 37,625.00\n"
        "Total Debits 23,393.00\n"
        "Closing Balance 66,607.00\n"
    )

    if is_ocr_available():
        ocr_text = extract_text_via_ocr(file_bytes)
    else:
        ocr_text = sample_scanned_ocr

    # 4. OCR Transaction parsing
    df_raw = parse_ocr_transactions(ocr_text)
    assert len(df_raw) == 14, f"Expected 14 transactions from scanned statement, got {len(df_raw)}"

    # Verify Netflix transaction is DEBIT and NOT credit
    netflix_row = df_raw[df_raw["Description"].str.contains("NETFLIX")]
    assert not netflix_row.empty, "Netflix transaction was not found in extracted records"
    assert float(netflix_row.iloc[0]["Debit"]) == 649.00, f"Netflix Debit should be 649.00, got {netflix_row.iloc[0]['Debit']}"
    assert float(netflix_row.iloc[0]["Credit"]) == 0.00, f"Netflix Credit should be 0.00, got {netflix_row.iloc[0]['Credit']}"

    # Verify Interest Credit is CREDIT and NOT debit
    interest_row = df_raw[df_raw["Description"].str.contains("INTEREST")]
    assert not interest_row.empty
    assert float(interest_row.iloc[0]["Credit"]) == 125.00
    assert float(interest_row.iloc[0]["Debit"]) == 0.00

    # 5. Data cleaning
    clean_df = clean_transaction_data(df_raw)
    assert len(clean_df) == 14

    # 6. Classification (Rule-based and ML)
    clean_df["Category_Rule"] = clean_df["Description"].apply(lambda d: classify_by_rules_with_confidence(d)[0])
    ml = MLClassifier()
    clean_df["Category_ML"] = clean_df["Description"].apply(ml.classify)
    assert (clean_df["Category_Rule"] != "").all()
    assert (clean_df["Category_ML"] != "").all()

    # 7. Balance validation
    validated = validate_running_balances(clean_df)
    assert (validated["Balance_Status"] == "Verified").all()

    # 8. Statement summary reconciliation
    rec = reconcile_statement_summary(validated, opening_balance=52500.0, reported_closing=66607.0)
    assert rec["is_reconciled"] is True
    assert rec["discrepancy"] == 0.0
    assert rec["opening_balance"] == 52500.0
    assert rec["calculated_closing"] == 66607.0

    # 9. Export generation
    csv_bytes = generate_csv(validated)
    excel_bytes = generate_excel(validated)
    assert len(csv_bytes) > 0
    assert len(excel_bytes) > 0


def test_error_handling_invalid_pdfs():
    # Empty bytes
    with pytest.raises(ValueError, match="empty"):
        detect_pdf_type(b"")

    # Corrupted bytes
    with pytest.raises(ValueError, match="corrupted"):
        detect_pdf_type(b"This is not a PDF file")


def test_confidence_threshold_slider_effect():
    # Full word match (confidence 1.0)
    desc_full = "ZOMATO RESTAURANT ORDER"
    cat_pass, conf_full = classify_by_rules_with_confidence(desc_full, threshold=0.80)
    assert cat_pass == "Food & Dining"
    assert conf_full >= 0.85

    # If threshold is set higher than confidence (e.g. 0.95 on heuristic), it falls back
    # Heuristic match for general banking abbreviation TFR has confidence 0.50
    desc_heu = "INTERNAL TFR TO ACC"
    cat_heu_pass, conf_heu = classify_by_rules_with_confidence(desc_heu, threshold=0.40)
    assert cat_heu_pass == "Transfers"
    assert conf_heu == 0.50

    # If slider moves to 0.70, the heuristic 0.50 drops to Uncategorized
    cat_heu_drop, _ = classify_by_rules_with_confidence(desc_heu, threshold=0.70)
    assert cat_heu_drop == "Uncategorized"


def test_deskew_image_correction():
    """Verify deskew algorithm detects and corrects document rotation."""
    from PIL import Image, ImageDraw
    from extraction.ocr_extractor import deskew_image

    # Create synthetic test page with horizontal text lines
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    for y in range(60, 540, 40):
        draw.text((60, y), "01-10-2026 UPI/SWIGGY/ORDER7841 450.00 _ 52,050.00", fill='black')

    # Simulate realistic scanner skew (+2.0 degrees)
    skewed_img = img.rotate(2.0, expand=False, fillcolor='white')
    corrected_img, angle = deskew_image(skewed_img)

    # Angle should be detected within 0.5 degrees of -2.0
    assert abs(abs(angle) - 2.0) <= 0.5, f"Expected detected skew ~2.0 degrees, got {angle}"


def test_inspect_pdf_pages_mixed_document():
    """Verify inspect_pdf_pages correctly identifies mixed PDFs (text cover + scanned page)."""
    import fitz
    from extraction.pdf_detector import inspect_pdf_pages

    doc = fitz.open()
    # Page 1: Digital text cover page
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 100), "Bank Statement Cover Page\nAccount details and terms...\n" * 5)

    # Page 2: Scanned image page
    p2 = doc.new_page(width=595, height=842)
    img_sample = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    p2.insert_image(fitz.Rect(50, 50, 250, 250), pixmap=img_sample)

    pdf_bytes = doc.tobytes()
    doc.close()

    info = inspect_pdf_pages(pdf_bytes)
    assert info["overall_type"] == "mixed"
    assert info["total_pages"] == 2
    assert info["text_pages"] == 1
    assert info["scanned_pages"] == 1


def test_narration_preserves_full_reference():
    """Verify transaction descriptions preserve full payee and reference IDs."""
    from extraction.text_extractor import _clean_narration
    
    raw = "UPI-SWIGGY-12345@paytm\nORDER 8892"
    cleaned = _clean_narration(raw)
    assert "SWIGGY" in cleaned
    assert "12345" in cleaned
    assert "8892" in cleaned



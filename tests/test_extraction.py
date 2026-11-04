import pytest
from extraction.transaction_parser import parse_account_details, parse_transactions

def test_parse_account_details():
    sample_text = "Account No: 1234567890\nIFSC: SBIN0001234\nName: John Doe"
    details = parse_account_details(sample_text)
    assert details["Account Number"] == "1234567890"
    assert details["IFSC Code"] == "SBIN0001234"
    assert "John" in details["Account Name"]

def test_parse_account_details_masked():
    sample_text = (
        "Sapphire Demo Bank\n"
        "Account Holder Aarav Mehta Statement Period 01 Oct 2026\n"
        "Account Number XXXXXX1234 (dummy) Account Type Savings\n"
        "IFSC SAPH0000001 (fictional) Currency INR\n"
    )
    details = parse_account_details(sample_text)
    assert details["Account Number"] == "XXXXXX1234"
    assert details["IFSC Code"] == "SAPH0000001"
    assert "Aarav Mehta" in details["Account Name"]
    assert "Sapphire" in details["Bank Name"]

def test_parse_account_details_hdfc():
    sample_text = (
        "MR RAHUL VERMA\n"
        "Account No : 50100456789012 SAVINGS\n"
        "RTGS/NEFT IFSC: HDFC0000048 MICR:380240003\n"
    )
    details = parse_account_details(sample_text)
    assert details["Account Number"] == "50100456789012"
    assert details["IFSC Code"] == "HDFC0000048"
    assert "RAHUL" in details["Account Name"]
    assert details["Bank Name"] == "HDFC Bank"

def test_parse_account_details_sbi():
    sample_text = (
        "Mr. Rajesh Kumar\n"
        "Branch Name : VASTRAPUR\n"
        "Account Number : 30992084658\n"
        "IFSC Code : SBIN0060447\n"
    )
    details = parse_account_details(sample_text)
    assert details["Account Number"] == "30992084658"
    assert details["IFSC Code"] == "SBIN0060447"
    assert details["Account Name"] == "Mr. Rajesh Kumar"
    assert details["Bank Name"] == "State Bank of India"

def test_parse_transactions_multi_format_dates():
    """Verify parsing handles DD-MM-YYYY, YYYY-MM-DD, and DD Mon YYYY dates."""
    sample_text = (
        "01 Oct 2026 UPI/SWIGGY/ORDER7841 450.00 - 52,050.00\n"
        "2026-10-02 SALARY CREDIT/ACME DEMO - 35,000.00 87,050.00\n"
        "03/10/2026 UPI/UBER/TRIP2918 320.00 - 86,730.00\n"
    )
    df = parse_transactions(sample_text)
    assert len(df) == 3
    assert df.iloc[0]["Date"] == "01 Oct 2026"
    assert df.iloc[0]["Description"] == "UPI/SWIGGY/ORDER7841"
    assert df.iloc[0]["Debit"] == "450.00"
    
    assert df.iloc[1]["Date"] == "2026-10-02"
    assert df.iloc[1]["Credit"] == "35000.00"
    
    assert df.iloc[2]["Date"] == "03/10/2026"
    assert df.iloc[2]["Debit"] == "320.00"

def test_parse_transactions_with_dashes_and_underscores():
    """Verify parser handles em-dash, hyphens, and OCR underscores."""
    sample_text = (
        "01-10-2026 UPI/SWIGGY/ORDER7841 450.00 \u2014 52,050.00\n"
        "02-10-2026 SALARY CREDIT/ACME DEMO PVT LTD _ 35,000.00 87,050.00\n"
        "14-10-2026 ATM CASH WITHDRAWAL 3,000.00 - 66,201.00\n"
    )
    df = parse_transactions(sample_text)
    assert len(df) == 3
    assert df.iloc[0]["Debit"] == "450.00"
    assert df.iloc[0]["Credit"] == "0.0"
    assert df.iloc[1]["Debit"] == "0.0"
    assert df.iloc[1]["Credit"] == "35000.00"
    assert df.iloc[2]["Balance"] == "66201.00"

import pytest
import pandas as pd
import io
import openpyxl
from export.export_data import generate_csv, generate_excel

def test_generate_csv():
    df = pd.DataFrame({
        'Date': ['2026-10-01'],
        'Description': ['Test Zomato'],
        'Debit': [150.0],
        'Credit': [0.0],
        'Balance': [1000.0],
        'Category': ['Food & Dining']
    })
    csv_bytes = generate_csv(df)
    assert isinstance(csv_bytes, bytes)
    text = csv_bytes.decode('utf-8')
    assert 'Test Zomato' in text
    assert 'Food & Dining' in text

def test_generate_excel_with_metadata_summary():
    df = pd.DataFrame({
        'Date': ['2026-10-01', '2026-10-02'],
        'Description': ['Test Amazon', 'Salary'],
        'Debit': [999.0, 0.0],
        'Credit': [0.0, 50000.0],
        'Balance': [9001.0, 59001.0],
        'Category': ['Shopping', 'Income']
    })
    account_details = {
        "Account Name": "Aarav Mehta",
        "Account Number": "123405001234",
        "IFSC Code": "ICIC0001234",
        "Bank Name": "ICICI Bank"
    }
    reconciliation = {
        "opening_balance": 10000.0,
        "total_debits": 999.0,
        "total_credits": 50000.0,
        "net_flow": 49001.0,
        "calculated_closing": 59001.0,
        "reported_closing": 59001.0,
        "is_reconciled": True
    }
    excel_bytes = generate_excel(df, account_details=account_details, reconciliation=reconciliation)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0
    
    # Verify openpyxl can read both sheets
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert 'Account Summary' in wb.sheetnames
    assert 'Transactions' in wb.sheetnames

    sheet_summary = wb['Account Summary']
    # Row 1 is header ['Property', 'Details']
    # Row 2 is Section header ['Account Information', None]
    # Row 3 is ['Bank Name', 'ICICI Bank']
    assert sheet_summary['B3'].value == 'ICICI Bank'
    assert sheet_summary['B4'].value == 'Aarav Mehta'

    sheet_txns = wb['Transactions']
    assert sheet_txns['B2'].value == 'Test Amazon'
    assert sheet_txns['F2'].value == 'Shopping'

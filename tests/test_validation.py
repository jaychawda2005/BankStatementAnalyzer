import pytest
import pandas as pd
from processing.balance_validator import validate_running_balances, reconcile_statement_summary
from processing.data_cleaner import clean_transaction_data

def test_validate_running_balances_clean():
    data = {
        'Date': ['2026-10-01', '2026-10-02', '2026-10-03'],
        'Description': ['Transaction 1', 'Transaction 2', 'Transaction 3'],
        'Debit': [100.0, 0.0, 50.0],
        'Credit': [0.0, 500.0, 0.0],
        'Balance': [900.0, 1400.0, 1350.0]
    }
    df = pd.DataFrame(data)
    result = validate_running_balances(df, opening_balance=1000.0)
    
    assert 'Balance_Status' in result.columns
    assert (result['Balance_Status'] == 'Verified').all()
    assert result['Discrepancy'].sum() == 0.0

def test_validate_running_balances_with_mismatch():
    data = {
        'Date': ['2026-10-01', '2026-10-02'],
        'Description': ['Txn 1', 'Txn 2'],
        'Debit': [100.0, 200.0],
        'Credit': [0.0, 0.0],
        'Balance': [1000.0, 750.0]  # Expected: 800.0, actual: 750.0 (Diff 50.0)
    }
    df = pd.DataFrame(data)
    result = validate_running_balances(df)
    
    assert result.iloc[1]['Balance_Status'] != 'Verified'
    assert 'Mismatch' in result.iloc[1]['Balance_Status']
    assert result.iloc[1]['Discrepancy'] == 50.0

def test_preserves_multiple_identical_transactions_same_day():
    """Verify clean_transaction_data preserves identical amounts on the same day with different balances."""
    data = {
        'Date': ['2026-10-01', '2026-10-01'],
        'Description': ['SWIGGY ORDER', 'SWIGGY ORDER'],
        'Debit': [450.0, 450.0],
        'Credit': [0.0, 0.0],
        'Balance': [52050.0, 51600.0]  # First order drops bal to 52050, second to 51600
    }
    df = pd.DataFrame(data)
    cleaned = clean_transaction_data(df)
    assert len(cleaned) == 2, "Second identical transaction on the same day must NOT be dropped!"
    
    validated = validate_running_balances(cleaned, opening_balance=52500.0)
    assert (validated['Balance_Status'] == 'Verified').all()

def test_reconcile_statement_summary():
    data = {
        'Date': ['2026-10-01', '2026-10-02'],
        'Description': ['Txn 1', 'Txn 2'],
        'Debit': [100.0, 200.0],
        'Credit': [500.0, 0.0],
        'Balance': [1400.0, 1200.0]
    }
    df = pd.DataFrame(data)
    summary = reconcile_statement_summary(df, opening_balance=1000.0, reported_closing=1200.0)
    
    assert summary['is_reconciled'] is True
    assert summary['total_debits'] == 300.0
    assert summary['total_credits'] == 500.0
    assert summary['calculated_closing'] == 1200.0
    assert summary['discrepancy'] == 0.0
    assert summary['opening_balance_source'] == "Explicitly Extracted from Document"

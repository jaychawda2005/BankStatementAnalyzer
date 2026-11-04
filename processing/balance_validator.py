import pandas as pd
import numpy as np

def validate_running_balances(df, opening_balance=None, tolerance=0.05):
    """
    Validates row-by-row transaction math:
    Balance[i] == Balance[i-1] - Debit[i] + Credit[i]
    
    If opening_balance is provided, Row 0 is also independently validated:
    Balance[0] == Opening_Balance - Debit[0] + Credit[0]
    
    Adds:
      - 'Expected_Balance': calculated balance based on previous balance & transaction
      - 'Balance_Status': 'Verified' if difference <= tolerance, else 'Mismatch'
      - 'Discrepancy': numerical difference
    """
    if df.empty or 'Balance' not in df.columns:
        return df

    df = df.copy()
    
    # Ensure numeric
    debits = pd.to_numeric(df['Debit'], errors='coerce').fillna(0.0).values
    credits = pd.to_numeric(df['Credit'], errors='coerce').fillna(0.0).values
    balances = pd.to_numeric(df['Balance'], errors='coerce').fillna(0.0).values

    n = len(df)
    expected_balances = []
    statuses = []
    discrepancies = []

    for i in range(n):
        if i == 0:
            if opening_balance is not None:
                expected = round(float(opening_balance) - debits[0] + credits[0], 2)
                diff = round(abs(balances[0] - expected), 2)
            else:
                expected = balances[0]
                diff = 0.0
        else:
            prev_bal = balances[i - 1]
            cur_bal = balances[i]
            expected = round(prev_bal - debits[i] + credits[i], 2)
            diff = round(abs(cur_bal - expected), 2)

        expected_balances.append(expected)
        discrepancies.append(diff)

        if diff <= tolerance:
            statuses.append("Verified")
        else:
            statuses.append(f"Mismatch (Diff: ₹{diff:,.2f})")

    df['Expected_Balance'] = expected_balances
    df['Balance_Status'] = statuses
    df['Discrepancy'] = discrepancies

    return df


def reconcile_statement_summary(df, opening_balance=None, reported_closing=None, tolerance=0.05):
    """
    Reconciles overall statement totals:
    Calculated Closing = Opening Balance - Total Debits + Total Credits
    
    Tracks whether Opening Balance was explicitly provided in the document
    or inferred from transaction records, preventing circular verification assumptions.
    """
    if df.empty:
        return {
            "total_debits": 0.0,
            "total_credits": 0.0,
            "net_flow": 0.0,
            "opening_balance": 0.0,
            "calculated_closing": 0.0,
            "reported_closing": 0.0,
            "is_reconciled": True,
            "mismatch_count": 0,
            "discrepancy": 0.0,
            "opening_balance_source": "None"
        }

    total_debits = round(float(pd.to_numeric(df['Debit'], errors='coerce').fillna(0.0).sum()), 2)
    total_credits = round(float(pd.to_numeric(df['Credit'], errors='coerce').fillna(0.0).sum()), 2)
    net_flow = round(total_credits - total_debits, 2)

    first_row = df.iloc[0]
    first_bal = float(first_row.get('Balance', 0.0))
    first_dr = float(first_row.get('Debit', 0.0))
    first_cr = float(first_row.get('Credit', 0.0))

    if opening_balance is None:
        inferred_opening = round(first_bal + first_dr - first_cr, 2)
        opening_source = "Inferred from First Row"
    else:
        inferred_opening = round(float(opening_balance), 2)
        opening_source = "Explicitly Extracted from Document"

    last_row = df.iloc[-1]
    last_bal = float(last_row.get('Balance', 0.0))

    if reported_closing is None:
        target_closing = last_bal
        closing_source = "Derived from Last Transaction Balance"
    else:
        target_closing = round(float(reported_closing), 2)
        closing_source = "Explicitly Extracted from Document"

    calculated_closing = round(inferred_opening - total_debits + total_credits, 2)
    discrepancy = round(abs(calculated_closing - target_closing), 2)
    is_reconciled = discrepancy <= tolerance

    # Count row-level mismatches if column exists
    mismatch_count = 0
    if 'Balance_Status' in df.columns:
        mismatch_count = int((df['Balance_Status'] != 'Verified').sum())

    return {
        "total_debits": total_debits,
        "total_credits": total_credits,
        "net_flow": net_flow,
        "opening_balance": inferred_opening,
        "calculated_closing": calculated_closing,
        "reported_closing": target_closing,
        "is_reconciled": is_reconciled,
        "mismatch_count": mismatch_count,
        "discrepancy": discrepancy,
        "opening_balance_source": opening_source,
        "closing_balance_source": closing_source
    }

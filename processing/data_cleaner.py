import pandas as pd

def clean_transaction_data(df):
    """
    Standardize transaction data:
    - Standardize Date to YYYY-MM-DD
    - Convert Debit, Credit, and Balance to numeric floats
    - Deduplicate only strictly identical rows including Balance
      (preserving legitimate multiple identical transactions on the same day)
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # Convert dates to standard datetime format (handles DD-MM-YYYY, YYYY-MM-DD, DD Mon YYYY, etc.)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Convert amounts to numeric floats
    for col in ['Debit', 'Credit', 'Balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    # Remove only exact identical rows including Balance
    # (e.g. repeated page headers or multi-page table overlap)
    # Does NOT delete legitimate multiple transactions of the same amount on the same day
    cols_to_check = [c for c in ['Date', 'Description', 'Debit', 'Credit', 'Balance'] if c in df.columns]
    if cols_to_check:
        df = df.drop_duplicates(subset=cols_to_check).reset_index(drop=True)
    
    return df

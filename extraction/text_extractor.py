import pdfplumber
import re

DATE_PATTERNS = [
    r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$',             # 01-10-2026, 1/10/26, 01.10.2026
    r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$',             # 2026-10-01, 2026/10/01
    r'^\d{1,2}[-\s/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s/]\d{2,4}$',  # 01 Oct 2026, 01-Oct-26
    r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}$'         # Oct 01, 2026
]
DATE_RE = re.compile('|'.join(DATE_PATTERNS), re.IGNORECASE)
AMT_RE = re.compile(r'[\d,]+\.\d{2}')

_TXN_START = re.compile(
    r'^(UPI-|NEFT|IMPS|POS |ACH |ATM |RTGS|INT |CHQ |BIL |EMI |ECS |CASH |DEP |WDL )',
    re.IGNORECASE
)

def extract_text_from_pdf(file_bytes):
    """Extract raw text across all pages in a PDF."""
    text = ""
    try:
        with pdfplumber.open(file_bytes) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text


def _clean_amt(v):
    """Convert raw cell value to decimal string; placeholders become 0.00."""
    if not v or str(v).strip() in ('', '-', '—', '–', '_', '~', 'None'):
        return '0.00'
    m = AMT_RE.search(str(v))
    return m.group(0).replace(',', '') if m else '0.00'


def _clean_narration(narr):
    """
    Clean whitespace and linebreaks while preserving the complete original narration.
    Retains reference IDs, payee details, and UPI/NEFT transaction information.
    """
    if not narr:
        return 'Transaction'
    lines = [l.strip() for l in str(narr).split('\n') if l.strip()]
    if not lines:
        return 'Transaction'
    text = ' '.join(lines)
    cleaned = text.strip(" -—–|\t\r\n")
    return cleaned if cleaned else 'Transaction'


def _find_column_indices(header_row):
    """Dynamically determine column indexes from a table header row."""
    col_map = {"date": None, "desc": None, "debit": None, "credit": None, "balance": None}
    for idx, cell in enumerate(header_row):
        if not cell:
            continue
        c = str(cell).strip().lower().replace('\n', ' ')
        if col_map["date"] is None and any(k in c for k in ['tran date', 'txn date', 'value date', 'post date', 'date']):
            col_map["date"] = idx
        elif any(k in c for k in ['balance', 'closing bal', 'net bal']):
            col_map["balance"] = idx
        elif any(k in c for k in ['debit', 'withdrawal', 'dr amt', 'dr.', 'dr']):
            col_map["debit"] = idx
        elif any(k in c for k in ['credit', 'deposit', 'cr amt', 'cr.', 'cr']):
            col_map["credit"] = idx
        elif col_map["desc"] is None and any(k in c for k in ['particular', 'detail', 'narration', 'description', 'remarks', 'transaction']):
            col_map["desc"] = idx
    return col_map


def extract_tables_from_pdf(file_bytes):
    """
    Unified multi-bank table extractor.
    Handles:
      1. Packed multi-transaction cell tables (HDFC style)
      2. Row-by-row standard tabular tables (SBI, ICICI, Axis, etc.)
    Returns list of dicts: Date, Description, Debit, Credit, Balance.
    """
    try:
        with pdfplumber.open(file_bytes) as pdf:
            # Check whether table uses packed multi-line cells (HDFC)
            is_packed = False
            for page in pdf.pages:
                tables = page.extract_tables()
                for t in tables:
                    for r in t:
                        if r and r[0] and '\n' in str(r[0]).strip():
                            first_line = str(r[0]).strip().split('\n')[0].strip()
                            if DATE_RE.match(first_line):
                                is_packed = True
                                break
                    if is_packed:
                        break
                if is_packed:
                    break

            if is_packed:
                return _extract_hdfc_style(pdf)
            else:
                return _extract_standard_table(pdf)
    except Exception as e:
        print(f"Table extraction error: {e}")
    return []


def extract_transactions_from_page(page, page_text=""):
    """
    Extract transactions from a single pdfplumber page.
    Attempts table extraction first, then falls back to regex parser on page_text.
    """
    txns = []
    tables = page.extract_tables()
    for t in tables:
        active_col_map = None
        for r in t:
            if not r or len(r) < 4:
                continue

            # Check if this row is a header row
            row_str = ' '.join(str(c or '').lower() for c in r)
            if any(k in row_str for k in ['date', 'txn', 'tran']) and any(k in row_str for k in ['balance', 'debit', 'credit', 'particular', 'description']):
                header_map = _find_column_indices(r)
                if header_map["date"] is not None and header_map["balance"] is not None:
                    active_col_map = header_map
                    continue

            # Determine date column
            d_col = active_col_map["date"] if active_col_map and active_col_map["date"] is not None else None
            if d_col is not None and d_col < len(r) and r[d_col] and DATE_RE.match(str(r[d_col]).strip()):
                date_val = str(r[d_col]).strip()
            elif r[0] and DATE_RE.match(str(r[0]).strip()):
                d_col = 0
                date_val = str(r[0]).strip()
            elif len(r) > 1 and r[1] and DATE_RE.match(str(r[1]).strip()):
                d_col = 1
                date_val = str(r[1]).strip()
            else:
                continue

            # Map columns using header map if available
            if active_col_map and active_col_map["balance"] is not None and active_col_map["desc"] is not None:
                desc_idx = active_col_map["desc"]
                details = r[desc_idx] if desc_idx < len(r) else "Transaction"
                debit_idx = active_col_map["debit"]
                debit = _clean_amt(r[debit_idx]) if debit_idx is not None and debit_idx < len(r) else "0.00"
                credit_idx = active_col_map["credit"]
                credit = _clean_amt(r[credit_idx]) if credit_idx is not None and credit_idx < len(r) else "0.00"
                bal_idx = active_col_map["balance"]
                bal = _clean_amt(r[bal_idx]) if bal_idx < len(r) else "0.00"
            elif len(r) == 7:
                # SBI format: [Value Date, Post Date, Details, Ref, Debit, Credit, Balance]
                details = r[2]
                debit = _clean_amt(r[4])
                credit = _clean_amt(r[5])
                bal = _clean_amt(r[6])
            elif len(r) == 6:
                details = r[d_col + 1]
                debit = _clean_amt(r[3])
                credit = _clean_amt(r[4])
                bal = _clean_amt(r[5])
            elif len(r) == 5:
                details = r[1]
                debit = _clean_amt(r[2])
                credit = _clean_amt(r[3])
                bal = _clean_amt(r[4])
            else:
                details = r[d_col + 1]
                nums = [_clean_amt(c) for c in r[-3:]]
                debit, credit, bal = nums[0], nums[1], nums[2]

            txns.append({
                "Date": date_val,
                "Description": _clean_narration(details),
                "Debit": debit,
                "Credit": credit,
                "Balance": bal
            })

    if not txns and page_text:
        from extraction.transaction_parser import parse_transactions
        parsed_df = parse_transactions(page_text)
        if not parsed_df.empty:
            txns = parsed_df.to_dict(orient="records")

    return txns


def _extract_standard_table(pdf):
    """Parse standard row-by-row bank tables (SBI, ICICI, Axis, etc.)."""
    txns = []
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        txns.extend(extract_transactions_from_page(page, page_text))
    return txns


def _extract_hdfc_style(pdf):
    """Parse HDFC-format packed tables across all pages."""
    full_text = '\n'.join(p.extract_text() or '' for p in pdf.pages)

    opening_balance = None
    ob_match = re.search(
        r'OpeningBalance\s+DrCount\s+CrCount\s+Debits\s+Credits\s+ClosingBal\s+([\d,]+\.\d{2})',
        full_text
    )
    if ob_match:
        opening_balance = float(ob_match.group(1).replace(',', ''))
    else:
        m = re.search(r'Opening\s*Balance[\s:]*([\d,]+\.\d{2})', full_text, re.IGNORECASE)
        if m:
            opening_balance = float(m.group(1).replace(',', ''))

    all_dates = []
    all_narrations_raw = []
    all_debits = []
    all_credits = []
    all_balances = []

    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                if not row or not row[0]:
                    continue
                first_cell = str(row[0]).strip().split('\n')[0].strip()
                if not DATE_RE.match(first_cell):
                    continue

                page_dates = [d.strip() for d in (row[0] or '').split('\n') if DATE_RE.match(d.strip())]
                raw_lines = (row[1] or '').split('\n')
                
                # Check column count
                if len(row) >= 7:
                    page_debits = [b.strip() for b in (row[4] or '').split('\n') if b.strip()]
                    page_credits = [b.strip() for b in (row[5] or '').split('\n') if b.strip()]
                    page_bals = [b.strip() for b in (row[6] or '').split('\n') if AMT_RE.search(b.strip())]
                else:
                    page_debits = []
                    page_credits = []
                    page_bals = [b.strip() for b in (row[-1] or '').split('\n') if AMT_RE.search(b.strip())]

                for line in raw_lines:
                    s = line.strip()
                    if not s:
                        continue
                    if _TXN_START.match(s):
                        all_narrations_raw.append(s)
                    else:
                        if all_narrations_raw:
                            all_narrations_raw[-1] += ' ' + s
                        else:
                            all_narrations_raw.append(s)

                all_dates.extend(page_dates)
                all_debits.extend(page_debits)
                all_credits.extend(page_credits)
                all_balances.extend(page_bals)

    n = len(all_dates)
    if n == 0:
        return []

    transactions = []
    prev_bal = opening_balance if opening_balance is not None else (float(all_balances[0].replace(',', '')) if all_balances else 0.0)

    for i in range(n):
        date = all_dates[i]
        raw_desc = all_narrations_raw[i] if i < len(all_narrations_raw) else 'Transaction'
        desc = _clean_narration(raw_desc)
        bal_str = all_balances[i] if i < len(all_balances) else "0.00"
        bal = float(bal_str.replace(',', ''))

        deb_raw = all_debits[i] if i < len(all_debits) else ""
        cred_raw = all_credits[i] if i < len(all_credits) else ""

        deb_cleaned = _clean_amt(deb_raw)
        cred_cleaned = _clean_amt(cred_raw)

        # If columns were empty or dashes, fall back to balance difference
        if float(deb_cleaned) == 0.0 and float(cred_cleaned) == 0.0 and prev_bal is not None:
            diff = round(bal - prev_bal, 2)
            if diff < 0:
                debit = f"{abs(diff):.2f}"
                credit = "0.00"
            else:
                debit = "0.00"
                credit = f"{diff:.2f}"
        else:
            debit = deb_cleaned
            credit = cred_cleaned

        transactions.append({
            "Date": date,
            "Description": desc,
            "Debit": debit,
            "Credit": credit,
            "Balance": f"{bal:.2f}"
        })
        prev_bal = bal

    return transactions

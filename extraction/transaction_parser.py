import re
import pandas as pd

def parse_account_details(text):
    details = {
        "Account Name": "Not Found",
        "Account Number": "Not Found",
        "IFSC Code": "Not Found",
        "Bank Name": "Unknown Bank"
    }
    
    # 1. IFSC Code: handles 'IFSC: ...', 'IFSC Code : ...', 'RTGS/NEFT IFSC: ...'
    ifsc_match = re.search(r'(?i)IFSC(?:\s*Code)?[\s:]*([A-Z]{4}0[A-Z0-9]{6})', text)
    if ifsc_match:
        details["IFSC Code"] = ifsc_match.group(1).upper()
        
    # 2. Bank Name via IFSC prefix
    ifsc_prefixes = {
        "HDFC": "HDFC Bank",
        "SBIN": "State Bank of India",
        "ICIC": "ICICI Bank",
        "UTIB": "Axis Bank",
        "KKBK": "Kotak Mahindra Bank",
        "BARB": "Bank of Baroda",
        "PUNB": "Punjab National Bank",
        "CNRB": "Canara Bank",
        "UBIN": "Union Bank of India",
        "IDFB": "IDFC First Bank",
        "YESB": "Yes Bank",
        "INDB": "IndusInd Bank",
    }
    if details["IFSC Code"] != "Not Found":
        prefix = details["IFSC Code"][:4]
        if prefix in ifsc_prefixes:
            details["Bank Name"] = ifsc_prefixes[prefix]

    if details["Bank Name"] == "Unknown Bank":
        for line in text.split('\n')[:15]:
            cleaned = line.strip()
            if re.search(r'(?i)\bbank\b', cleaned) and not re.search(r'(?i)(sample|demo document|statement|page|branch|address)', cleaned):
                details["Bank Name"] = cleaned
                break
    
    # 3. Account Number: pure digits (9-18) OR masked format like XXXXXX1234
    acc_num_match = re.search(
        r'(?i)account\s*(?:no|number)[\.:\s]*([A-Z0-9X\*\-]{6,18})', text
    )
    if acc_num_match:
        details["Account Number"] = acc_num_match.group(1).strip()
        
    # 4. Account Holder / Customer Name / Name
    title_match = re.search(r'(?m)^(?:Mr\.|Mrs\.|Ms\.|MR|MRS|MS|DR)\.?\s+([A-Za-z ]+)$', text)
    if title_match and len(title_match.group(0).strip()) > 3:
        details["Account Name"] = title_match.group(0).strip()
    else:
        holder_match = re.search(
            r'(?i)(?:account\s*holder(?:\s*name)?|customer\s*name|account\s*name)[\s:]+([A-Za-z\s\.]+?)(?=\s+(?:statement|account|ifsc|branch|date|currency)|\n|$)',
            text
        )
        if holder_match and len(holder_match.group(1).strip()) > 2:
            details["Account Name"] = holder_match.group(1).strip()
        else:
            gen_name = re.search(
                r'(?i)(?<!branch\s)(?<!bank\s)(?<!product\s)(?<!nominee\s)\bname[\s:]+([A-Za-z\s\.]+?)(?=\s+(?:statement|account|ifsc|branch|date|currency)|\n|$)',
                text
            )
            if gen_name and len(gen_name.group(1).strip()) > 2:
                details["Account Name"] = gen_name.group(1).strip()
        
    return details


def _is_placeholder(val):
    """Check if token is an empty or placeholder symbol (_, —, –, -, ~, 0)."""
    if not val:
        return True
    s = str(val).strip()
    return bool(re.fullmatch(r'[_—–\-~]+|\b0\b|\b0\.00\b', s))


def _clean_amount(val):
    """Convert a raw amount string to a clean decimal string. Placeholders mean zero."""
    if not val or _is_placeholder(val):
        return "0.0"
    m = re.search(r'[\d,]+\.\d{2}', str(val))
    if m:
        return m.group(0).replace(',', '').strip()
    m_int = re.search(r'\b\d+\b', str(val))
    if m_int:
        return f"{float(m_int.group(0)):.2f}"
    return "0.0"


DATE_PATTERN_STR = (
    r'(?:'
    r'[0-3]?[0-9][-/.\s][0-1]?[0-9][-/.\s](?:19|20)?[0-9]{2}|'
    r'(?:19|20)[0-9]{2}[-/.\s][0-1]?[0-9][-/.\s][0-3]?[0-9]|'
    r'[0-3]?[0-9][-\s/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s/](?:19|20)?[0-9]{2}|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[0-3]?[0-9],?\s+(?:19|20)?[0-9]{2}'
    r')'
)


def parse_ocr_transactions(text):
    """
    Robust transaction parser for OCR text from scanned statements.
    - Supports multiple date formats (DD-MM-YYYY, YYYY-MM-DD, DD Mon YYYY).
    - Captures amount and placeholder tokens including underscores (_), dashes (—, –, -), and tildes (~).
    - Uses strict word boundaries instead of substring matching (e.g. \\bCR\\b vs 'SUBSCRIPTION').
    - Leverages running balance difference to disambiguate single-amount lines.
    - Preserves transaction descriptions without dropping words.
    """
    transactions = []
    running_balance = None
    open_m = re.search(r'(?i)opening\s*balance[\s:]*(?:[A-Z\$\u20b9\s]*)([\d,]+\.\d{2})', text)
    if open_m:
        running_balance = float(open_m.group(1).replace(',', ''))

    for raw_line in text.split('\n'):
        line = raw_line.replace('|', ' ').replace('■', ' ').replace('•', ' ').strip()
        if not line:
            continue

        # Look for date token across supported formats
        date_match = re.search(r'\b(' + DATE_PATTERN_STR + r')\b', line, re.IGNORECASE)
        if not date_match:
            continue

        date_str = date_match.group(1).strip()
        rest = line[date_match.end():].strip()

        # Extract amount and placeholder tokens
        tokens = re.findall(r'([\d,]+\.\d{2}|[_—–\-~]+|\b0\b)', rest)
        if len(tokens) < 2:
            continue

        balance_token = tokens[-1]
        if not re.search(r'\d', balance_token):
            continue

        balance_val = float(_clean_amount(balance_token))

        # Determine Debit and Credit tokens
        if len(tokens) >= 3:
            cand_debit = tokens[-3]
            cand_credit = tokens[-2]

            if _is_placeholder(cand_debit) and not _is_placeholder(cand_credit):
                debit_token, credit_token = "0.0", cand_credit
            elif not _is_placeholder(cand_debit) and _is_placeholder(cand_credit):
                debit_token, credit_token = cand_debit, "0.0"
            elif not _is_placeholder(cand_debit) and not _is_placeholder(cand_credit):
                debit_token, credit_token = cand_debit, cand_credit
            else:
                debit_token, credit_token = "0.0", "0.0"
        else:
            # Missing amount column (only 2 tokens: amount & balance)
            amt_token = tokens[-2]
            amt_val = float(_clean_amount(amt_token))

            # 1. Use running balance difference if available
            if running_balance is not None and abs(balance_val - running_balance) > 0.01:
                diff = balance_val - running_balance
                if diff > 0.01:
                    debit_token, credit_token = "0.0", amt_token
                else:
                    debit_token, credit_token = amt_token, "0.0"
            else:
                # 2. Strict word boundary keyword matching (avoid false matches like 'SUBSCRIPTION')
                is_credit_kw = bool(re.search(
                    r'\b(CR|CREDIT|CREDITED|DEP|DEPOSIT|RECEIVED|REFUND|SALARY|INTEREST|DIVIDEND|REVERSAL)\b',
                    rest,
                    re.IGNORECASE
                ))
                is_debit_kw = bool(re.search(
                    r'\b(DR|DEBIT|DEBITED|PAID|PAYMENT|WITHDRAWAL|ATM|PURCHASE|POS|BILL|FEE|CHARGES|SUBSCRIPTION|ORDER)\b',
                    rest,
                    re.IGNORECASE
                ))
                if is_credit_kw and not is_debit_kw:
                    debit_token, credit_token = "0.0", amt_token
                else:
                    debit_token, credit_token = amt_token, "0.0"

        # Extract description: text preceding the amount column tokens
        col_tokens = tokens[-3:] if len(tokens) >= 3 else tokens[-2:]
        desc_part = rest
        first_tok = col_tokens[0]
        tok_idx = desc_part.rfind(first_tok)
        desc = desc_part[:tok_idx].strip() if tok_idx != -1 else desc_part.strip()
        desc = re.sub(r'[\s_—–\-~]+$', '', desc).strip()

        transactions.append({
            "Date": date_str,
            "Description": desc or "Transaction",
            "Debit": _clean_amount(debit_token),
            "Credit": _clean_amount(credit_token),
            "Balance": _clean_amount(balance_token)
        })
        running_balance = balance_val

    return pd.DataFrame(transactions)


def parse_transactions(text):
    """
    Primary transaction parser: tries standard regex first (supporting multiple date formats:
    DD-MM-YYYY, YYYY-MM-DD, DD Mon YYYY), then automatically falls back to parse_ocr_transactions.
    """
    pattern = re.compile(
        r'\b(' + DATE_PATTERN_STR + r')\s+'        # Date (any standard format)
        r'(.+?)\s+'                                # Description
        r'([\d,]+\.\d{2}|[\u2014\u2013\-_~]+)\s+' # Debit OR placeholder
        r'([\d,]+\.\d{2}|[\u2014\u2013\-_~]+)\s+' # Credit OR placeholder
        r'([\d,]+\.\d{2})',                        # Balance
        re.IGNORECASE
    )
    
    transactions = []
    for line in text.split('\n'):
        match = pattern.search(line)
        if match:
            date, desc, debit, credit, balance = match.groups()
            transactions.append({
                "Date": date.strip(),
                "Description": desc.strip(),
                "Debit": _clean_amount(debit),
                "Credit": _clean_amount(credit),
                "Balance": _clean_amount(balance)
            })
            
    df = pd.DataFrame(transactions)
    if df.empty:
        # Fall back to robust OCR parser
        df = parse_ocr_transactions(text)

    return df

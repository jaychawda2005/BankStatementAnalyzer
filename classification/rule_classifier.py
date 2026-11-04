import re

# Comprehensive dictionary of keyword patterns using strict word boundaries
# Maps categories to lists of keywords / phrases.
CATEGORY_RULES = {
    'Food & Dining': [
        'ZOMATO', 'SWIGGY', 'MCDONALDS', 'DOMINOS', 'STARBUCKS', 'BURGER KING',
        'SUBWAY', 'PIZZA HUT', 'KFC', 'HALDIRAM', 'BARBEQUE NATION', 'CHAAYOS',
        'CAFE COFFEE DAY', 'INSTAMART', 'BAKERY', 'RESTAURANT', 'DINE', 'FOOD',
        'PIZZA', 'BURGER', 'COFFEE', 'CAFE', 'TEA', 'SWEETS'
    ],
    'Shopping': [
        'RELIANCE FRESH', 'RELIANCE SMART', 'RELIANCE', 'AMAZON', 'FLIPKART', 'MYNTRA',
        'DMART', 'BIGBASKET', 'ZEPTO', 'BLINKIT', 'ZARA', 'NYKAA', 'CROMA', 'TATA CLIQ',
        'AJIO', 'DECATHLON', 'SUPERMARKET', 'GROCERY', 'GROCERIES', 'MART', 'APPAREL',
        'CLOTHING', 'MALL', 'RETAIL STORE'
    ],
    'Fuel': [
        'SHELL', 'HPCL', 'BPCL', 'IOCL', 'INDIAN OIL', 'PETROL', 'DIESEL',
        'FUEL', 'CNG', 'FASTAG', 'TOLL PLAZA', 'REPAIR', 'GARAGE'
    ],
    'Transportation': [
        'UBER', 'OLA', 'RAPIDO', 'IRCTC', 'INDIGO', 'AIR INDIA', 'VISTARA',
        'SPICEJET', 'RED BUS', 'METRO RAIL', 'FLIGHT', 'AIRLINES', 'RAILWAY',
        'CAB', 'TAXI', 'AUTO FARE', 'MAKEMYTRIP', 'YATRA'
    ],
    'Utilities': [
        'ELECTRICITY', 'POWER', 'UGVCL', 'BESCOM', 'MAHAVITARAN', 'TATA POWER',
        'AIRTEL', 'JIO', 'VODAFONE', 'BROADBAND', 'FIBERNET', 'ACT FIBERNET',
        'GAS BILL', 'INDANE', 'BHARAT GAS', 'WATER TAX', 'WATER BILL', 'DTH',
        'TATA PLAY', 'DISH TV', 'PIPED GAS'
    ],
    'Entertainment': [
        'NETFLIX', 'SPOTIFY', 'BOOKMYSHOW', 'HOTSTAR', 'DISNEY', 'APPLE MEDIA',
        'YOUTUBE PREMIUM', 'SONY LIV', 'AMAZON PRIME', 'PVR CINEMAS', 'INOX',
        'PLAYSTATION', 'STEAM GAMES', 'GAANA', 'ZEE5', 'AUDIBLE', 'SUBSCRIPTION'
    ],
    'Healthcare': [
        'APOLLO PHARMACY', 'MEDPLUS', 'NETMEDS', 'PHARMEASY', '1MG',
        'HEALTHPLUS', 'LAL PATHLABS', 'METROPOLIS', 'HOSPITAL', 'CLINIC',
        'PHARMACY', 'CHEMIST', 'DOCTOR', 'DENTAL', 'MEDICINE', 'DIAGNOSTICS',
        'PATHOLOGY', 'HEALTHCARE'
    ],
    'Loans & EMI': [
        'EMI', 'LOAN', 'HOUSING LOAN', 'HOME LOAN', 'CAR LOAN', 'AUTO LOAN',
        'PERSONAL LOAN', 'EDUCATION LOAN', 'BAJAJ FINANCE', 'HDFC LOAN',
        'SBI FINANCE', 'INSTALLMENT', 'MORTGAGE', 'FINANCE'
    ],
    'Insurance': [
        'LIC OF INDIA', 'LIC', 'HDFC LIFE', 'ICICI PRU', 'STAR HEALTH', 'MAX LIFE',
        'NEW INDIA ASSURANCE', 'TATA AIG', 'CARE HEALTH', 'INSURANCE',
        'POLICY PREMIUM', 'HEALTH COVER'
    ],
    'Taxes': [
        'INCOME TAX', 'ADVANCE TAX', 'GST', 'TDS', 'CHALLAN 280',
        'PROPERTY TAX', 'MUNICIPAL TAX', 'COMMERCIAL TAX', 'TAX PAYMENT'
    ],
    'Investment': [
        'MUTUAL FUND', 'SIP', 'ZERODHA', 'GROWW', 'UPSTOX', 'NIPPON INDIA',
        'PPF', 'NPS', 'SECURITIES', 'DEMAT', 'SHARE BROKING', 'GOLD BOND',
        'SGB', 'STOCK BROKER', 'INVESTMENT'
    ],
    'Rent': [
        'RENT PAYMENT', 'RENT', 'HOUSE RENT', 'FLAT RENT', 'OFFICE RENT', 'SHOP RENT',
        'SOCIETY MAINTENANCE', 'APARTMENT RENT', 'LEASE PAYMENT', 'HOSTEL RENT',
        'MAINTENANCE CHARGES'
    ],
    'Bank Charges': [
        'SMS ALERT', 'ANNUAL FEE', 'DEBIT CARD FEE', 'MINIMUM BALANCE',
        'ATM CHARGE', 'BOUNCE CHARGE', 'PENALTY', 'SERVICE CHARGE',
        'CONSOLIDATED CHARGES', 'LEDGER FOLIO', 'BANK CHARGES'
    ],
    'ATM Withdrawal': [
        'ATM CASH', 'CASH WDL', 'CASH WITHDRAWAL', 'ATM WDL', 'SELF WITHDRAWAL',
        'CARDLESS CASH', 'ATM'
    ],
    'Income': [
        'SALARY', 'INTEREST CREDIT', 'DIVIDEND', 'BONUS', 'STIPEND',
        'CHQ DEP', 'CHEQUE DEPOSIT', 'CHQ DEPOSIT', 'REFUND', 'CASHBACK',
        'REIMBURSEMENT', 'PAYROLL', 'DIRECT DEPOSIT'
    ],
    'Transfers': [
        'UPI', 'NEFT', 'IMPS', 'RTGS', 'PHONEPE', 'PAYTM', 'GOOGLE PAY',
        'GPAY', 'ONE97', 'CRED', 'LAZYPAY', 'TRANSFER', 'REMITTANCE'
    ]
}

# Compile strict word boundary regex patterns
# Separate specific category rules from generic payment channel rules (Transfers)
# This guarantees that merchant/category terms strictly take priority over channel tokens.
_SPECIFIC_RULES = []
_CHANNEL_RULES = []

for category, keywords in CATEGORY_RULES.items():
    rule_list = _CHANNEL_RULES if category == 'Transfers' else _SPECIFIC_RULES
    for kw in keywords:
        pattern = r'\b' + re.escape(kw).replace(r'\ ', r'[\s/-]+') + r'\b'
        rule_list.append((re.compile(pattern, re.IGNORECASE), kw, category))

# Sort rules by keyword length descending within their respective buckets
_SPECIFIC_RULES.sort(key=lambda x: len(x[1]), reverse=True)
_CHANNEL_RULES.sort(key=lambda x: len(x[1]), reverse=True)


def classify_by_rules_with_confidence(description, debit=0.0, credit=0.0, threshold=0.50):
    """
    Classify a transaction description using strict word-boundary heuristic rules
    and direction awareness (Debit vs Credit).
    
    Specific merchant and category keywords strictly take priority over generic
    payment-channel keywords (UPI, NEFT, IMPS, RTGS, POS, ACH, CHQ).
    
    Parameters:
        description (str): Transaction narrative text.
        debit (float): Debit amount (if known).
        credit (float): Credit amount (if known).
        threshold (float): Minimum confidence threshold (default 0.50).
        
    Returns:
        tuple: (category: str, confidence: float)
    """
    desc = str(description).strip()
    if not desc or desc.upper() == "TRANSACTION":
        return "Uncategorized", 0.0

    desc_upper = desc.upper()
    has_debit = float(debit or 0.0) > 0.0
    has_credit = float(credit or 0.0) > 0.0

    # 1. Direction-aware pre-checks
    if has_debit and not has_credit:
        if re.search(r'\b(HOME LOAN|CAR LOAN|PERSONAL LOAN|LOAN|EMI|MORTGAGE)\b', desc_upper):
            return "Loans & EMI", 1.0
        if re.search(r'\b(INTEREST\s*(?:PAID|DEBIT|CHARGE)|LOAN\s*INTEREST)\b', desc_upper):
            return "Loans & EMI", 0.95

    # Refund / reimbursement / cheque deposit on credit transactions
    if has_credit and not has_debit:
        if re.search(r'\b(REFUND|REVERSAL|CASHBACK|REIMBURSEMENT)\b', desc_upper):
            conf = 0.95
            if conf >= threshold:
                return "Income", conf
            return "Uncategorized", conf
        if re.search(r'\b(CHQ DEP|CHEQUE DEPOSIT|CHQ DEPOSIT)\b', desc_upper):
            conf = 0.95
            if conf >= threshold:
                return "Income", conf
            return "Uncategorized", conf

    # If REFUND appears in narrative even without explicit credit indicator, treat as Income
    if re.search(r'\b(REFUND|REIMBURSEMENT)\b', desc_upper) and not has_debit:
        conf = 0.95
        if conf >= threshold:
            return "Income", conf
        return "Uncategorized", conf

    # 2. Priority Phase 1: Match Specific Merchant & Category Rules
    for regex, kw, category in _SPECIFIC_RULES:
        if regex.search(desc):
            # Direction check: Income rules (Salary, Interest, Dividend)
            # must not apply if this is strictly a debit transaction
            if category == 'Income' and has_debit and not has_credit:
                if 'INTEREST' in kw:
                    return "Bank Charges", 0.85
                continue

            confidence = 1.0 if ' ' in kw or len(kw) >= 5 else 0.85
            if confidence < threshold:
                return "Uncategorized", round(confidence, 2)
            return category, round(confidence, 2)

    # 3. Priority Phase 2: Fall back to Generic Channel Rules (Transfers)
    for regex, kw, category in _CHANNEL_RULES:
        if regex.search(desc):
            confidence = 0.70
            if confidence < threshold:
                return "Uncategorized", round(confidence, 2)
            return category, round(confidence, 2)

    # 4. Fallback heuristic matching for banking codes (UPI, TFR, DEP, WDL)
    if re.search(r'\b(UPI|TFR|TRANSFER|NEFT|IMPS|RTGS)\b', desc_upper):
        conf = 0.50
        if conf >= threshold:
            return "Transfers", conf
        return "Uncategorized", conf

    return "Uncategorized", 0.0


def classify_by_rules(description):
    """Convenience function returning only category string."""
    cat, _ = classify_by_rules_with_confidence(description)
    return cat

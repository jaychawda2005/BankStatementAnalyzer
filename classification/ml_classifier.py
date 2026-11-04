import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import plotly.express as px

# Comprehensive synthetic banking training dataset (16 categories, 403 realistic samples)
# Privacy-safe: zero personal names or real customer account records
# Extended with realistic Indian bank-narration channel noise (UPI/, POS/, NEFT/, IMPS/, ACH D-)
TRAINING_DATA = {
    "Description": [
        # Food & Dining (26)
        "ZOMATO ONLINE ORDER", "SWIGGY FOOD DELIVERY", "MCDONALDS BURGER", "DOMINOS PIZZA OUTLET",
        "STARBUCKS COFFEE CAFE", "CAFE COFFEE DAY", "SUBWAY SANDWICH", "BURGER KING MEAL",
        "HALDIRAM SWEETS RESTAURANT", "BARBEQUE NATION DINNER", "CHAAYOS TEA CAFE", "PIZZA HUT DINE IN",
        "KFC CHICKEN OUTLET", "BAKERY SWEETS PURCHASE", "RESTAURANT FOOD BILL",
        "UPI/SWIGGY/ORDER7841", "UPI-ZOMATO-9921@paytm", "POS/DOMINOS PIZZA", "UPI/MCDONALDS/PUN",
        "POS/STARBUCKS CAFE", "UPI/BURGERKING/901", "UPI/CHAAYOS/MUMBAI", "POS/HALDIRAM/REST",
        "UPI/KFC/FOODORDER", "POS/SUBWAY/SANDWICH", "UPI/BARBEQUE NATION/DINE",
        
        # Shopping (27)
        "AMAZON INDIA SHOPPING", "FLIPKART INTERNET RETAIL", "MYNTRA CLOTHING FASHION", "DMART SUPERMARKET STORE",
        "BIGBASKET GROCERY ORDER", "ZEPTO QUICK GROCERY", "BLINKIT QUICK COMMERCE", "RELIANCE SMART BAZAAR",
        "ZARA APPAREL STORE", "NYKAA COSMETICS BEAUTY", "CROMA ELECTRONICS STORE", "TATA CLIQ ONLINE",
        "AJIO FASHION RETAIL", "DECATHLON SPORTS GEAR", "MORE RETAIL SUPERMARKET",
        "UPI/AMAZON/ORDER 55320", "POS/DMART/AHMEDABAD", "UPI-FLIPKART-PAY", "POS/RELIANCE FRESH",
        "POS/RELIANCE SMART", "UPI/MYNTRA/FASHION", "POS/CROMA ELECTRONICS", "UPI/ZEPTO/GROCERY",
        "UPI/BLINKIT/STORE", "POS/ZARA/RETAIL", "UPI/BIGBASKET/ORDER", "MOB BKG UPI PHONEPE GROCERY",
        
        # Transportation (25)
        "UBER RIDE TRIP", "OLA CABS CITY AUTO", "RAPIDO BIKE TAXI", "IRCTC TRAIN TICKET",
        "INDIGO AIRLINES BOOKING", "METRO RAIL SMART CARD", "AIR INDIA FLIGHT TICKET", "VISTARA AIRLINE SEAT",
        "RED BUS TICKET BOOKING", "AUTO RICKSHAW FARE", "RAILWAY RESERVATION COUNTER", "AIRPORT CAB SERVICE",
        "MAKEMYTRIP TRAVEL BOOKING", "YATRA AIRWAYS TRAVEL", "CHALO BUS PASS SMART",
        "UPI/UBER/TRIP 2918", "UPI-OLA CABS-AUTO", "UPI/RAPIDO/RIDE", "IRCTC/TICKET/BOOKING",
        "NEFT/INDIGO AIRLINES", "UPI/METRO RECHARGE", "POS/AIR INDIA COUNTER", "UPI/VISTARA AIRLINE",
        "UPI/RED BUS TICKET", "POS/MAKEMYTRIP TRIP",
        
        # Fuel (25)
        "SHELL PETROL PUMP FUEL", "HPCL AUTO FUEL RECHARGE", "BPCL PETROL STATION", "INDIAN OIL FUEL PUMP",
        "IOCL PETROL DIESEL REFUEL", "FASTAG TOLL PLAZA HIGHWAY", "HIGHWAY TOLL CHARGES", "CNG AUTO GAS FILLING",
        "TOTAL ENERGIES FUEL PUMP", "PETROL DIESEL REFUEL PUMP", "HP AUTO CARE FUEL", "BHARAT PETROLEUM DISPENSER",
        "AUTO FUEL FILLING STATION", "RELIANCE PETROLEUM PUMP", "ESSAR PETROL FUEL STATION",
        "POS/SHELL PETROL PUMP", "POS/HPCL FUEL RECHARGE", "POS/BPCL AUTO FUEL", "POS/INDIAN OIL CORP",
        "POS/IOCL DIESEL PUMP", "NETC/FASTAG/TOLL PLAZA", "UPI/HP AUTO CARE FUEL", "POS/PETROL FILLING STN",
        "POS/CNG GAS STATION", "UPI/BHARAT PETROLEUM",
        
        # Utilities (25)
        "ELECTRICITY BILL PAYMENT UGVCL", "BESCOM POWER UTILITY", "AIRTEL MOBILE PREPAID RECHARGE", "JIO FIBER BROADBAND BILL",
        "TATA PLAY DTH RECHARGE", "ADANI GAS PIPELINE", "MUNICIPAL WATER TAX BILL", "VODAFONE IDEA POSTPAID BILL",
        "ACT FIBERNET BROADBAND", "MAHAVITARAN ELECTRICITY BILL", "INDANE GAS CYLINDER REFILL", "BROADBAND INTERNET SERVICE",
        "DTH CABLE TV SUBSCRIPTION", "PIPED NATURAL GAS UTILITY", "SOLAR POWER UTILITY BILL",
        "ELECTRICITY BILL/UGVCL", "UPI/BESCOM POWER/BILL", "UPI/AIRTEL PREPAID/REC", "UPI/JIO FIBER BROADBAND",
        "NEFT/TATA POWER BILL", "UPI/TATA PLAY DTH REC", "UPI/ADANI GAS PIPELINE", "UPI/MUNICIPAL WATER TAX",
        "NEFT/MAHAVITARAN POWER", "UPI/ACT FIBERNET/NET",
        
        # Entertainment (25)
        "NETFLIX MONTHLY SUBSCRIPTION", "SPOTIFY MUSIC STREAMING", "BOOKMYSHOW MOVIE TICKETS", "DISNEY HOTSTAR ANNUAL",
        "APPLE MEDIA SERVICES", "YOUTUBE PREMIUM FAMILY", "SONY LIV OTT STREAMING", "AMAZON PRIME VIDEO",
        "PVR CINEMAS MULTIPLEX", "INOX MOVIES BOX OFFICE", "PLAYSTATION STORE GAMES", "GAANA MUSIC STREAMING",
        "ZEE5 PREMIUM ANNUAL", "AUDIBLE AUDIOBOOKS", "STEAM DIGITAL VIDEO GAMES",
        "NETFLIX SUBSCRIPTION", "ACH D- SPOTIFY PREMIUM", "UPI/BOOKMYSHOW/TICKETS", "UPI/HOTSTAR ANNUAL SUB",
        "APPLE.COM/BILL MEDIA", "UPI/YOUTUBE PREMIUM", "POS/PVR CINEMAS", "POS/INOX MULTIPLEX",
        "ACH D- SONY LIV OTT", "UPI/PLAYSTATION/STORE",
        
        # Healthcare (25)
        "APOLLO PHARMACY MEDICINES", "MEDPLUS CHEMIST STORE", "NETMEDS HEALTHCARE MEDICINES", "HEALTHPLUS DIAGNOSTIC LAB",
        "DR LAL PATHLABS TEST", "MAX SUPER HOSPITAL BILL", "PHARMEASY ONLINE MEDICINE", "TATA 1MG PHARMACY ORDER",
        "DENTAL CARE CLINIC VISIT", "METROPOLIS DIAGNOSTICS LAB", "FORTIS HEALTHCARE CLINIC", "WELLNESS PHARMACY STORE",
        "EYE CLINIC CONSULTATION", "BLOOD TEST LAB FEES", "PEDIATRIC CLINIC HOSPITAL",
        "UPI/PHARMACY/HEALTHPLUS", "POS/APOLLO PHARMACY", "UPI/MEDPLUS CHEMIST", "UPI/NETMEDS MEDICINES",
        "POS/DR LAL PATHLABS", "NEFT/HOSPITAL TREATMENT", "UPI/PHARMEASY MEDICINE", "POS/METROPOLIS LAB",
        "UPI/1MG HEALTHCARE", "POS/FORTIS CLINIC",
        
        # Loans & EMI (25)
        "HOME LOAN EMI PAYMENT", "CAR LOAN INSTALLMENT", "PERSONAL LOAN REPAYMENT", "BAJAJ FINANCE EMI",
        "HDFC BANK LOAN EMI", "SBI HOME FINANCE EMI", "EDUCATION LOAN INSTALLMENT", "GOLD LOAN INTEREST PAYMENT",
        "TWO WHEELER LOAN EMI", "HOUSING FINANCE MONTHLY DUES", "VEHICLE FINANCE INSTALLMENT", "MORTGAGE LOAN PAYMENT",
        "LOAN REPAYMENT DEBIT", "COMMERCIAL VEHICLE EMI", "CONSUMER DURABLE LOAN EMI",
        "ACH D- HOME LOAN EMI", "ACH D- CAR LOAN EMI", "ACH D- BAJAJ FINANCE EMI", "NEFT/HDFC LOAN REPAY",
        "ACH D- PERSONAL LOAN", "ECS/SBI HOME FINANCE", "ACH D- TWO WHEELER EMI", "NEFT/LOAN INSTALLMENT",
        "ACH D- HOUSING FINANCE", "ACH D- VEHICLE LOAN",
        
        # Insurance (25)
        "LIC LIFE INSURANCE PREMIUM", "HDFC LIFE INSURANCE POLICY", "ICICI PRU LIFE PREMIUM", "STAR HEALTH INSURANCE RENEWAL",
        "MAX LIFE INSURANCE ANNUAL", "MOTOR VEHICLE INSURANCE", "CARE HEALTH INSURANCE COVER", "NEW INDIA ASSURANCE POLICY",
        "TATA AIG GENERAL INSURANCE", "TERM LIFE INSURANCE PAYMENT", "ACCIDENT INSURANCE PREMIUM", "HEALTH COVER POLICY RENEWAL",
        "BAJAJ ALLIANZ LIFE PREMIUM", "SBI LIFE INSURANCE POLICY", "GENERAL INSURANCE COVER",
        "ACH D- LIC OF INDIA", "ACH D- HDFC LIFE PREMIUM", "NEFT/STAR HEALTH INS", "ACH D- ICICI PRU LIFE",
        "ACH D- MAX LIFE POLICY", "NEFT/CARE HEALTH COVER", "ACH D- NEW INDIA ASSUR", "ACH D- TATA AIG INS",
        "ECS/LIFE INSURANCE PREM", "ACH D- POLICY PREMIUM",
        
        # Taxes (25)
        "INCOME TAX E-FILING PAYMENT", "ADVANCE TAX CHALLAN 280", "GST TAX PAYMENT GOVT", "MUNICIPAL PROPERTY TAX",
        "TDS TAX DEPOSIT GOVT", "SELF ASSESSMENT TAX 280", "MUNICIPAL CORPORATION TAX", "COMMERCIAL TAX DEPARTMENT",
        "CENTRAL BOARD DIRECT TAXES", "STATE REVENUE TAX PAYMENT", "PROPERTY TAX ASSESSMENT", "CHALLAN TAX PAYMENT GOVT",
        "CORPORATE TAX DEPOSIT", "PROFESSIONAL TAX PAYMENT", "ROAD TAX RTO PAYMENT",
        "CBDT/INCOME TAX/280", "GST/TAX PAYMENT/GOVT", "TDS/TAX DEDUCTED GOVT", "CHALLAN 280 TAX DEP",
        "NEFT/MUNICIPAL PROPERTY", "COMMERCIAL TAX DEPT", "ADVANCE TAX PAYMENT", "SELF ASSESSMENT TAX",
        "ROAD TAX RTO PAYMENT", "DIRECT TAXES E-FILING",
        
        # Investment (25)
        "SIP MUTUAL FUND INVESTMENT", "ZERODHA BROKING SECURITIES", "GROWW EQUITY INVESTMENTS", "NIPPON INDIA MUTUAL FUND",
        "PPF PUBLIC PROVIDENT FUND", "UPSTOX TRADING DEMAT", "SBI MUTUAL FUND SIP", "NATIONAL PENSION SYSTEM NPS",
        "UTI MUTUAL FUND SCHEME", "MIRAE ASSET MUTUAL FUND", "KOTAK MUTUAL FUND SIP", "SOVEREIGN GOLD BOND SGB",
        "STOCK PURCHASE BROKERAGE", "ICICI PRU MUTUAL FUND", "AXIS MUTUAL FUND SIP",
        "UPI/ZERODHA BROKING/DEP", "UPI/GROWW INVEST/MUTUAL", "ACH D- SIP MUTUAL FUND", "ACH D- NIPPON INDIA MF",
        "NEFT/UPSTOX TRADING", "NEFT/PPF CONTRIBUTION", "ACH D- SBI MUTUAL FUND", "ACH D- NPS TRUST CONTRIB",
        "ACH D- MIRAE ASSET MF", "UPI/KOTAK SECURITIES",
        
        # Rent (25)
        "HOUSE RENT PAYMENT OWNER", "MONTHLY FLAT RENT", "SOCIETY MAINTENANCE CHARGES", "OFFICE LEASE RENT",
        "HOSTEL ACCOMMODATION RENT", "APARTMENT RENT TRANSFER", "COMMERCIAL SPACE RENT", "RESIDENTIAL LEASE PAYMENT",
        "BUILDING MAINTENANCE DUES", "SHOP LEASE RENT OWNER", "COWORKING DESK RENT", "WAREHOUSE LEASE RENT",
        "PROPERTY LEASE RENT OWNER", "GUEST HOUSE RENT", "PG ACCOMMODATION MONTHLY RENT",
        "NEFT/RENT PAYMENT", "UPI/HOUSE RENT/OWNER", "NEFT/MONTHLY FLAT RENT", "UPI/SOCIETY MAINTENANCE",
        "NEFT/OFFICE LEASE RENT", "UPI/APARTMENT RENT", "NEFT/COMMERCIAL RENT", "UPI/HOSTEL RENT DUES",
        "NEFT/BUILDING MAINT", "UPI/PG MONTHLY RENT",
        
        # Bank Charges (25)
        "BANK SMS ALERT CHARGES", "ANNUAL DEBIT CARD CHARGES", "MINIMUM BALANCE PENALTY", "ATM TRANSACTION CHARGES",
        "CHEQUE BOUNCE PENALTY", "LEDGER FOLIO CHARGES", "ACCOUNT SERVICE CHARGES", "OVERDRAFT SERVICE FEE",
        "IMPS OUTWARD SERVICE CHARGE", "STOP CHEQUE PAYMENT FEE", "LOCKER RENT ANNUAL CHARGES", "CONSOLIDATED CHARGES",
        "CARD REPLACEMENT CHARGE", "FOREIGN CURRENCY MARKUP FEE", "INWARD REMITTANCE CHARGE",
        "BANK SMS ALERT CHARGE", "ANNUAL DEBIT CARD CHG", "MIN BALANCE PENALTY", "ATM USAGE CHARGES",
        "CHQ BOUNCE CHARGES", "LEDGER FOLIO CHARGES", "CONSOLIDATED CHARGES", "IMPS OUTWARD CHARGE",
        "ACCOUNT SERVICE FEE", "LOCKER RENT ANNUAL",
        
        # ATM Withdrawal (25)
        "ATM CASH WITHDRAWAL SBI", "CASH WDL BANK ATM", "HDFC ATM CASH WITHDRAWAL", "KOTAK ATM CASH WITHDRAWAL",
        "ATM CASH DISPENSER MACHINE", "SELF CASH WITHDRAWAL BRANCH", "AXIS BANK ATM CASH WDL", "ATM WITHDRAWAL TELLER",
        "CARDLESS CASH WITHDRAWAL", "BANK BRANCH CASH WDL", "ATM CASH WITHDRAWAL ICICI", "CASH DISPENSED ATM",
        "PUNJAB NATIONAL BANK ATM", "CANARA BANK ATM CASH", "UNION BANK ATM WDL",
        "ATM CASH WITHDRAWAL", "ATM CASH WDL SBI 891", "CASH WDL BANK ATM", "HDFC ATM CASH WDL",
        "ATM CASH DISPENSER", "AXIS ATM CASH WDL", "ATM WITHDRAWAL TELLER", "BRANCH CASH WDL SELF",
        "ICICI ATM CASH WDL", "PNB ATM CASH DISPENSE",
        
        # Transfers (25)
        "UPI TRANSFER TO BENEFICIARY", "NEFT TRANSFER TO ACCOUNT", "IMPS FUND TRANSFER RECEIVED", "PAYTM WALLET TOPUP",
        "PHONEPE PEER TRANSFER", "GOOGLE PAY UPI TRANSACTION", "CRED CREDIT CARD BILL PAYMENT", "LAZYPAY REPAYMENT ACCOUNT",
        "RTGS INTERBANK TRANSFER", "INTERNAL ACCOUNT TRANSFER", "BILLDESK PAYMENT GATEWAY", "RAZORPAY MERCHANT PAYMENT",
        "UPI PEER TO PEER TRANSFER", "IMPS OUTWARD REMITTANCE", "NEFT SETTLEMENT CLEARING",
        "UPI/TRANSFER/BENEFICIARY", "NEFT/TRANSFER/ACC", "IMPS/TRANSFER/P2P", "UPI/PAYTM WALLET/ADD",
        "UPI/PHONEPE/TRANSFER", "UPI/GOOGLEPAY/SEND", "CRED/CREDIT CARD BILL", "RTGS/INTERBANK/TRANSFER",
        "UPI/PAY TO CONTACT", "NEFT/FUND TRANSFER OUT",
        
        # Income (25)
        "MONTHLY SALARY CREDIT", "SALARY CREDIT ACME DEMO", "QUARTERLY SAVINGS INTEREST CREDIT", "DIVIDEND RECEIVED COMPANY",
        "ANNUAL BONUS CREDIT", "INCOME TAX REFUND CREDIT", "CONSULTING FEES RECEIVED", "FREELANCE PAYMENT CREDIT",
        "PAYROLL DIRECT DEPOSIT", "INTERNSHIP STIPEND CREDIT", "TRAVEL REIMBURSEMENT CLAIM", "CASHBACK REWARD CREDIT",
        "PROFIT SHARE DISTRIBUTION", "SAVINGS BANK INTEREST CREDIT", "FIXED DEPOSIT INTEREST CREDIT",
        "SALARY CREDIT/ACME DEMO PVT LTD", "MONTHLY SALARY CREDIT", "INTEREST CREDIT", "SAVINGS INTEREST CREDIT",
        "CHQ DEP/CLEARING", "IMPS RECEIVED/FRIEND", "IMPS/REFUND FLIPKART", "UPI/REFUND/AMAZON",
        "DIVIDEND CREDIT COMPANY", "ANNUAL BONUS CREDIT"
    ],
    "Category": [
        "Food & Dining"] * 26 + [
        "Shopping"] * 27 + [
        "Transportation"] * 25 + [
        "Fuel"] * 25 + [
        "Utilities"] * 25 + [
        "Entertainment"] * 25 + [
        "Healthcare"] * 25 + [
        "Loans & EMI"] * 25 + [
        "Insurance"] * 25 + [
        "Taxes"] * 25 + [
        "Investment"] * 25 + [
        "Rent"] * 25 + [
        "Bank Charges"] * 25 + [
        "ATM Withdrawal"] * 25 + [
        "Transfers"] * 25 + [
        "Income"] * 25
}


class MLClassifier:
    """
    Non-LLM Machine Learning Classifier using TF-IDF Vectorization
    (Word n-grams + Character n-grams FeatureUnion) and Logistic Regression.
    
    Trained on 403 clean, diverse synthetic banking narratives across 16 categories.
    Noise-resistant to typical banking prefixes (UPI/, POS/, NEFT/, IMPS/, ACH D-).
    Reports genuine 5-fold cross-validation metrics.
    """
    def __init__(self):
        self.categories = sorted(list(set(TRAINING_DATA["Category"])))
        feature_extractor = FeatureUnion([
            ('word', TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r'[A-Za-z0-9]+'
            )),
            ('char_wb', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                sublinear_tf=True
            ))
        ])
        
        self.model = make_pipeline(
            feature_extractor,
            LogisticRegression(
                C=3.0,
                max_iter=500,
                class_weight='balanced',
                random_state=42
            )
        )
        self.metrics = None
        self._cv_preds = None
        self._train_and_evaluate()

    def _train_and_evaluate(self):
        """Train model and compute genuine 5-Fold Stratified Cross-Validation metrics."""
        X = TRAINING_DATA["Description"]
        y = np.array(TRAINING_DATA["Category"])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_preds = cross_val_predict(self.model, X, y, cv=cv)
        self._cv_preds = cv_preds

        acc = accuracy_score(y, cv_preds)
        prec = precision_score(y, cv_preds, average='weighted', zero_division=0)
        rec = recall_score(y, cv_preds, average='weighted', zero_division=0)
        f1 = f1_score(y, cv_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y, cv_preds, labels=self.categories)

        self.metrics = {
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "confusion_matrix": cm,
            "classes": self.categories,
            "training_samples": len(X),
            "num_classes": len(self.categories),
            "report": classification_report(y, cv_preds, labels=self.categories, output_dict=True, zero_division=0)
        }

        # Fit model on all training data for production inference
        self.model.fit(X, y)
        self.is_trained = True

    def get_evaluation_metrics(self):
        """Return genuine cross-validation performance metrics dictionary."""
        return self.metrics

    def create_confusion_matrix_figure(self):
        """Generate an interactive Plotly confusion matrix heatmap."""
        if not self.metrics:
            return None

        cm = self.metrics["confusion_matrix"]
        labels = self.categories

        fig = px.imshow(
            cm,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
            title=f"5-Fold Cross-Validation Confusion Matrix (Accuracy: {self.metrics['accuracy']*100:.1f}%)"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            width=700,
            height=600,
            margin=dict(l=100, r=40, t=60, b=100)
        )
        return fig

    def classify_with_confidence(self, description, threshold=0.15):
        """
        Classify transaction with calibrated probability confidence.
        
        Parameters:
            description (str): Transaction narrative
            threshold (float): Minimum probability threshold (default 0.15).
                               Sub-threshold predictions fall back to 'Uncategorized'.
        """
        if not description or str(description).strip() == "" or str(description).strip().upper() == "TRANSACTION":
            return "Uncategorized", 0.0

        desc = str(description).strip()
        probs = self.model.predict_proba([desc])[0]
        max_idx = np.argmax(probs)
        raw_confidence = float(probs[max_idx])
        pred_label = self.model.classes_[max_idx]

        if raw_confidence < threshold:
            return "Uncategorized", round(raw_confidence, 2)

        return pred_label, round(raw_confidence, 2)

    def classify(self, description):
        """Simple classification interface returning category string."""
        cat, _ = self.classify_with_confidence(description)
        return cat

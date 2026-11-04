import pandas as pd
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

def generate_csv(df):
    """Generate clean UTF-8 encoded CSV of categorized transactions."""
    return df.to_csv(index=False).encode('utf-8')


def generate_excel(df, account_details=None, reconciliation=None):
    """
    Generate professional Excel workbook with:
    1. 'Account Summary' sheet: Account Holder, Number, IFSC, Bank, and Reconciliation Audit
    2. 'Transactions' sheet: Formatted records with auto-adjusted column widths and styling
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Create Account Summary Sheet if metadata is available
        if account_details or reconciliation:
            summary_rows = []
            if account_details:
                summary_rows.append(["Account Information", ""])
                summary_rows.append(["Bank Name", account_details.get("Bank Name", "Unknown Bank")])
                summary_rows.append(["Account Holder", account_details.get("Account Name", "Not Found")])
                summary_rows.append(["Account Number", account_details.get("Account Number", "Not Found")])
                summary_rows.append(["IFSC Code", account_details.get("IFSC Code", "Not Found")])
                summary_rows.append(["", ""])

            if reconciliation:
                summary_rows.append(["Balance Reconciliation Audit", ""])
                summary_rows.append(["Opening Balance", f"₹ {reconciliation.get('opening_balance', 0.0):,.2f}"])
                summary_rows.append(["Total Debits", f"₹ {reconciliation.get('total_debits', 0.0):,.2f}"])
                summary_rows.append(["Total Credits", f"₹ {reconciliation.get('total_credits', 0.0):,.2f}"])
                summary_rows.append(["Net Cash Flow", f"₹ {reconciliation.get('net_flow', 0.0):,.2f}"])
                summary_rows.append(["Calculated Closing", f"₹ {reconciliation.get('calculated_closing', 0.0):,.2f}"])
                summary_rows.append(["Reported Closing", f"₹ {reconciliation.get('reported_closing', 0.0):,.2f}"])
                reconciled_str = "Verified Reconciled (0 Mismatches)" if reconciliation.get("is_reconciled") else f"Discrepancy (₹ {reconciliation.get('discrepancy', 0.0):,.2f})"
                summary_rows.append(["Reconciliation Status", reconciled_str])
                if "opening_balance_source" in reconciliation:
                    summary_rows.append(["Opening Balance Source", reconciliation["opening_balance_source"]])

            summary_df = pd.DataFrame(summary_rows, columns=["Property", "Details"])
            summary_df.to_excel(writer, index=False, sheet_name="Account Summary")
            
            ws_summary = writer.sheets["Account Summary"]
            ws_summary.column_dimensions["A"].width = 28
            ws_summary.column_dimensions["B"].width = 38
            header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
            for cell in ws_summary[1]:
                cell.font = header_font
                cell.fill = header_fill

        # 2. Transactions Sheet
        df.to_excel(writer, index=False, sheet_name='Transactions')
        ws_txn = writer.sheets['Transactions']
        
        # Style transaction header
        th_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        th_fill = PatternFill(start_color="203764", end_color="203764", fill_type="solid")
        for cell in ws_txn[1]:
            cell.font = th_font
            cell.fill = th_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Auto-adjust column widths
        for col in ws_txn.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                val_str = str(cell.value or '')
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws_txn.column_dimensions[col_letter].width = max(max_len + 4, 12)

    return output.getvalue()

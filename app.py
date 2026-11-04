import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px

from extraction.pdf_detector import inspect_pdf_pages
from extraction.pipeline import extract_document
from classification.rule_classifier import classify_by_rules_with_confidence
from classification.ml_classifier import MLClassifier
from processing.data_cleaner import clean_transaction_data
from processing.balance_validator import validate_running_balances, reconcile_statement_summary
from export.export_data import generate_csv, generate_excel

# Page configuration
st.set_page_config(
    page_title="Bank Statement Processing & Classification", 
    page_icon="🏦", 
    layout="wide"
)

# Initialize ML model in session cache
@st.cache_resource
def get_ml_classifier():
    return MLClassifier()

ml_classifier = get_ml_classifier()

st.title("🏦 Bank Statement Processing & Classification System")
st.markdown("Automated non-LLM pipeline for multi-bank statement parsing, OCR text extraction, balance verification, and transaction categorization.")

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    classification_method = st.radio(
        "Classification Engine", 
        ("Rule-Based Heuristics", "Machine Learning (TF-IDF + Logistic Regression)")
    )
    
    st.markdown("---")
    default_thresh = 0.50 if classification_method == "Rule-Based Heuristics" else 0.15
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_thresh,
        step=0.05,
        help="Transactions classified with confidence below this threshold fall back to 'Uncategorized'."
    )
    
    st.markdown("---")
    st.subheader("🤖 Model Evaluation Metrics")
    metrics = ml_classifier.get_evaluation_metrics()
    if metrics:
        st.write(f"**Training Samples:** `{metrics['training_samples']}` (16 categories)")
        st.write(f"**CV Accuracy:** `{metrics['accuracy'] * 100:.1f}%`")
        st.write(f"**Weighted F1:** `{metrics['f1_weighted'] * 100:.1f}%`")
        st.write(f"**Weighted Precision:** `{metrics['precision_weighted'] * 100:.1f}%`")
        st.write(f"**Weighted Recall:** `{metrics['recall_weighted'] * 100:.1f}%`")
        
        with st.expander("📊 View Confusion Matrix"):
            cm_fig = ml_classifier.create_confusion_matrix_figure()
            if cm_fig:
                st.plotly_chart(cm_fig, use_container_width=True)
                
    st.markdown("---")
    st.info("Upload standard text-based, scanned, or mixed PDFs. Auto-routes between table parsing, text regex, and deskewed Tesseract OCR.")


def process_and_render_statement(file_obj, key_suffix="0"):
    """Process and render complete pipeline for a single statement file."""
    with st.spinner(f"Analyzing and parsing {file_obj.name}..."):
        file_bytes = file_obj.read()
        
        # 1. Page Inspection & Type Detection
        try:
            pdf_info = inspect_pdf_pages(file_bytes)
            pdf_type = pdf_info["overall_type"]
            st.success(
                f"Document detected as: **{pdf_type.upper()} PDF** "
                f"({pdf_info['total_pages']} pages: {pdf_info['text_pages']} text, {pdf_info['scanned_pages']} scanned)"
            )
        except ValueError as ve:
            st.error(f"❌ {str(ve)}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error reading PDF: {str(e)}")
            return
        
        # 2. Unified Document Extraction Pipeline (with Per-Page Routing)
        try:
            raw_df, raw_text, account_details, extraction_method = extract_document(file_bytes)
        except Exception as e:
            st.error(f"Error during document extraction: {str(e)}")
            return

        if "Warning:" in extraction_method:
            st.warning(f"⚠️ {extraction_method}")
        else:
            st.info(f"Extraction Pipeline: **{extraction_method}** — **{len(raw_df)} transactions** extracted.")

        if raw_df.empty:
            st.error("Could not extract tabular transactions. Please verify that the PDF follows standard bank statement formatting.")
            return

        # 3. Cleaning & Standardization (preserves real same-day duplicates with distinct balances)
        clean_df = clean_transaction_data(raw_df)

        # 4. Classification Engine with Confidence Scoring & Direction Awareness
        categories = []
        confidences = []
        
        for idx, row in clean_df.iterrows():
            desc = row["Description"]
            deb = float(row.get("Debit", 0.0) or 0.0)
            cred = float(row.get("Credit", 0.0) or 0.0)
            
            if classification_method == "Rule-Based Heuristics":
                cat, conf = classify_by_rules_with_confidence(
                    desc, debit=deb, credit=cred, threshold=confidence_threshold
                )
            else:
                cat, conf = ml_classifier.classify_with_confidence(
                    desc, threshold=confidence_threshold
                )
                
            categories.append(cat)
            confidences.append(f"{int(conf * 100)}%")
            
        clean_df["Category"] = categories
        clean_df["Confidence"] = confidences

        # 5. Extract Opening/Closing balances if explicitly stated in text
        open_m = re.search(r'(?i)opening\s*balance[\s:]*(?:[A-Z\$\u20b9\s]*)([\d,]+\.\d{2})', raw_text) if raw_text else None
        closing_m = re.search(r'(?i)closing\s*balance[\s:]*(?:[A-Z\$\u20b9\s]*)([\d,]+\.\d{2})', raw_text) if raw_text else None
        ob_val = float(open_m.group(1).replace(',', '')) if open_m else None
        cb_val = float(closing_m.group(1).replace(',', '')) if closing_m else None

        # 6. Balance Validation (Row-by-Row + Summary Audit)
        validated_df = validate_running_balances(clean_df, opening_balance=ob_val)
        reconciliation = reconcile_statement_summary(validated_df, opening_balance=ob_val, reported_closing=cb_val)

        # ---------------------------------------------------------
        # Account Details & Summary Metrics
        # ---------------------------------------------------------
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Account Metadata")
            st.json(account_details)
            
            total_debit = reconciliation["total_debits"]
            total_credit = reconciliation["total_credits"]
            net_flow = reconciliation["net_flow"]
            
            m1, m2 = st.columns(2)
            m1.metric("Total Debits", f"₹ {total_debit:,.2f}")
            m2.metric("Total Credits", f"₹ {total_credit:,.2f}")
            st.metric("Net Cash Flow", f"₹ {net_flow:,.2f}")
            
        with col2:
            st.subheader("⚖️ Balance Reconciliation Audit")
            if reconciliation["is_reconciled"] and reconciliation["mismatch_count"] == 0:
                st.success("✅ **Statement Fully Reconciled**: All transaction balances match mathematically.")
            else:
                st.warning(
                    f"⚠️ **Audit Notice**: {reconciliation['mismatch_count']} row discrepancy detected. "
                    f"Net variance: ₹ {reconciliation['discrepancy']:,.2f}"
                )
                
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Opening Balance", f"₹ {reconciliation['opening_balance']:,.2f}")
            r_col2.metric("Calculated Closing", f"₹ {reconciliation['calculated_closing']:,.2f}")
            r_col3.metric("Reported Closing", f"₹ {reconciliation['reported_closing']:,.2f}")
            
            st.caption(f"ℹ️ Opening Balance Source: **{reconciliation.get('opening_balance_source', 'N/A')}**")

        # ---------------------------------------------------------
        # Interactive Transaction Table with Manual Category Override
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("💳 Transaction Records (Interactive & Editable)")
        st.caption("You can click on any category in the table below to manually override it before downloading.")
        
        display_cols = ['Date', 'Description', 'Debit', 'Credit', 'Balance', 'Category', 'Confidence', 'Balance_Status']
        valid_cols = [c for c in display_cols if c in validated_df.columns]
        
        all_available_categories = [
            "Food & Dining", "Shopping", "Transportation", "Fuel", "Utilities",
            "Entertainment", "Healthcare", "Loans & EMI", "Insurance", "Taxes",
            "Investment", "Rent", "Bank Charges", "ATM Withdrawal", "Transfers",
            "Income", "Uncategorized"
        ]

        edited_df = st.data_editor(
            validated_df[valid_cols],
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category (Editable)",
                    help="Select to override category",
                    options=all_available_categories,
                    required=True
                ),
                "Date": st.column_config.TextColumn("Date", disabled=True),
                "Description": st.column_config.TextColumn("Description", disabled=True),
                "Debit": st.column_config.NumberColumn("Debit", format="₹ %.2f", disabled=True),
                "Credit": st.column_config.NumberColumn("Credit", format="₹ %.2f", disabled=True),
                "Balance": st.column_config.NumberColumn("Balance", format="₹ %.2f", disabled=True),
                "Confidence": st.column_config.TextColumn("Confidence", disabled=True),
                "Balance_Status": st.column_config.TextColumn("Status", disabled=True)
            },
            use_container_width=True,
            key=f"transaction_table_editor_{key_suffix}"
        )
        
        # ---------------------------------------------------------
        # Financial Visualizations
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Financial Analytics")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            expense_df = edited_df[edited_df["Debit"] > 0]
            if not expense_df.empty:
                fig_pie = px.pie(
                    expense_df, 
                    values='Debit', 
                    names='Category', 
                    title='Expense Breakdown by Category',
                    hole=0.3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No debit expense transactions available for pie chart.")
                
        with viz_col2:
            daily_df = edited_df.groupby('Date')[['Debit', 'Credit']].sum().reset_index()
            fig_bar = px.bar(
                daily_df, 
                x='Date', 
                y=['Debit', 'Credit'], 
                barmode='group', 
                title='Daily Cash Inflow vs Outflow',
                color_discrete_map={'Debit': '#EF553B', 'Credit': '#00CC96'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # ---------------------------------------------------------
        # Data Export (with Account Summary & Formatted Worksheets)
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📥 Export Processed Statement")
        exp_col1, exp_col2, _ = st.columns([1, 1, 2])
        
        export_df = edited_df[valid_cols]
        csv_data = generate_csv(export_df)
        excel_data = generate_excel(export_df, account_details=account_details, reconciliation=reconciliation)
        
        base_name = file_obj.name.rsplit('.', 1)[0]
        exp_col1.download_button(
            "📄 Download CSV", 
            csv_data, 
            f"{base_name}_processed.csv", 
            "text/csv",
            key=f"btn_csv_{key_suffix}"
        )
        exp_col2.download_button(
            "📊 Download Excel (with Summary Sheet)", 
            excel_data, 
            f"{base_name}_processed.xlsx", 
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"btn_excel_{key_suffix}"
        )


# ---------------------------------------------------------
# Document Upload & Processing Pipeline (Multi-File Support)
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Bank Statement(s) (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) == 1:
        process_and_render_statement(uploaded_files[0], key_suffix="single")
    else:
        st.info(f"📁 **{len(uploaded_files)} statements uploaded**. Switch tabs below to inspect each document:")
        tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files])
        for idx, tab in enumerate(tabs):
            with tab:
                process_and_render_statement(uploaded_files[idx], key_suffix=str(idx))

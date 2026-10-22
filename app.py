"""
app.py
======
Fraud Flagger - High-Risk Transaction Monitor.

This file provides the streamlined Streamlit user interface:
- Top-of-page data ingestion with CSV file uploader.
- Strict requirement: Analyzes and displays results ONLY after the user uploads a CSV file.
- Comprehensive details and dossier for High-Risk Flagged Transactions.
- No sidebar, no charts, no threshold sliders, and no sample datasets.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

import fraud_detection as fd


# ---------------------------------------------------------------------------
# Page Configuration & Styling (No Sidebar)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Flagger | High-Risk Transaction Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark Fintech Cybersecurity CSS Theme with Sidebar Removed
CUSTOM_CSS = """
<style>
    /* Remove and hide Streamlit sidebar completely */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {
        display: none !important;
    }
    
    .stApp {
        background-color: #0B1120;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Header Box */
    .brand-header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .brand-title {
        color: #EF4444;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: 0.04em;
    }
    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Ingestion Box at top */
    .ingestion-container {
        background: #111C35;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }

    /* High Flagged Card */
    .flagged-card {
        background: #151F38;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .flagged-card.locked-out {
        border-left-color: #EC4899;
        background: #1F172E;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid #EF4444;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-lockout {
        background: rgba(236, 72, 153, 0.2);
        color: #F472B6;
        border: 1px solid #EC4899;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin-left: 6px;
    }

    /* Summary Stat Chips */
    .stat-chip {
        background: #0F172A;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-chip-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .stat-chip-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-top: 4px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-header-box">
        <h1 class="brand-title">🛡️ FRAUD FLAGGER</h1>
        <p class="brand-subtitle">High-Risk Transaction Monitoring & Security Alert System</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Ingestion Area (Top of Page) - Upload Only
# ---------------------------------------------------------------------------
st.markdown('<div class="ingestion-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload Transaction CSV File",
    type=["csv"],
    help="Upload your transaction CSV. The system evaluates repeated PIN failures, unusual amounts, and rapid transaction frequency.",
)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Strict Gate: Show Results ONLY After User Uploads CSV File
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.info(
        "👆 **Awaiting CSV File:** Please upload a transaction CSV file above to run fraud detection and view high flagged transactions.",
        icon="📁",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Load & Validate Uploaded CSV
# ---------------------------------------------------------------------------
try:
    raw_df, load_warnings = fd.load_csv(uploaded_file)
except Exception as e:
    st.error(f"Error parsing uploaded CSV file: {str(e)}")
    st.stop()

if raw_df is None or raw_df.empty:
    st.error("The uploaded CSV file is empty or contains no readable transaction rows.")
    st.stop()

if load_warnings:
    with st.expander("ℹ️ CSV Column Mapping & Ingestion Notes", expanded=False):
        for w in load_warnings:
            st.caption(f"• {w}")


# ---------------------------------------------------------------------------
# Run Backend Analysis Engine
# ---------------------------------------------------------------------------
with st.spinner("Analyzing uploaded transactions against fraud criteria..."):
    analysis = fd.analyze_transactions(
        df=raw_df,
        pin_threshold=3,
        amount_multiplier=3.0,
        window_minutes=5,
        max_transactions=5,
    )

if not analysis["is_valid"]:
    st.error("### ❌ Data Validation Error")
    for err in analysis["warnings"]:
        st.error(f"- {err}")
    st.stop()

processed_df: pd.DataFrame = analysis["df"]

# Filter strictly for HIGH FLAGGED TRANSACTIONS (Risk Level == High, Score >= 50, or Lockout)
high_flagged_df = processed_df[
    (processed_df["risk_level"] == "High") | (processed_df.get("auth_lockout", False))
].copy()

total_ingested = len(processed_df)
high_count = len(high_flagged_df)
total_high_volume = high_flagged_df["amount"].sum() if high_count > 0 else 0.0
locked_count = int(high_flagged_df.get("auth_lockout", pd.Series(False)).sum())

# Status summary banner
st.caption(
    f"**Analyzed File:** `{uploaded_file.name}` | "
    f"**Total Ingested Rows:** {total_ingested:,} | "
    f"**High-Risk Flagged Count:** {high_count:,}"
)

# Stat Chips
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.markdown(
        f"""
        <div class="stat-chip">
            <div class="stat-chip-label">High-Risk Transactions</div>
            <div class="stat-chip-val" style="color: #EF4444;">{high_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col2:
    st.markdown(
        f"""
        <div class="stat-chip">
            <div class="stat-chip-label">High-Risk Exposed Volume</div>
            <div class="stat-chip-val" style="color: #F8FAFC;">₹{total_high_volume:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col3:
    st.markdown(
        f"""
        <div class="stat-chip">
            <div class="stat-chip-label">Authentication Lockouts</div>
            <div class="stat-chip-val" style="color: #EC4899;">{locked_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat_col4:
    rate = round((high_count / total_ingested * 100), 1) if total_ingested > 0 else 0.0
    st.markdown(
        f"""
        <div class="stat-chip">
            <div class="stat-chip-label">High-Risk Rate</div>
            <div class="stat-chip-val" style="color: #38BDF8;">{rate}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if high_flagged_df.empty:
    st.success("✅ **Clean Security Status:** No high-risk flagged transactions were detected in the uploaded dataset.")
    st.stop()

# Sort high flagged transactions by risk score descending
high_flagged_df = high_flagged_df.sort_values(by="risk_score", ascending=False)


# ---------------------------------------------------------------------------
# Section 1: Summary Table of High Flagged Transactions
# ---------------------------------------------------------------------------
st.markdown("### 🚨 **High Flagged Transactions Summary Table**")

table_df = high_flagged_df.copy()
table_df["triggered_rules_text"] = table_df["triggered_criteria"].apply(
    lambda x: ", ".join(x) if isinstance(x, list) and x else "None"
)

column_config = {
    "transaction_id": st.column_config.TextColumn("Transaction ID", width="small"),
    "user_id": st.column_config.TextColumn("User ID", width="small"),
    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
    "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
    "pin_attempts": st.column_config.NumberColumn("PIN Retries", format="%d"),
    "frequency_in_window": st.column_config.NumberColumn("Window Freq", format="%d"),
    "risk_score": st.column_config.ProgressColumn(
        "Risk Score",
        help="Risk score out of 100",
        format="%d",
        min_value=0,
        max_value=100,
    ),
    "risk_level": st.column_config.TextColumn("Risk Tier"),
    "auth_lockout": st.column_config.CheckboxColumn("Account Locked"),
    "triggered_rules_text": st.column_config.TextColumn("Triggered Criteria", width="medium"),
    "recommended_action": st.column_config.TextColumn("Recommended Action", width="large"),
}

columns_to_show = [
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "pin_attempts",
    "frequency_in_window",
    "risk_score",
    "risk_level",
    "auth_lockout",
    "triggered_rules_text",
    "recommended_action",
]

st.dataframe(
    table_df[columns_to_show],
    column_config=column_config,
    use_container_width=True,
    hide_index=True,
)

st.write("")




# ---------------------------------------------------------------------------
# CSV Export for High Flagged Transactions
# ---------------------------------------------------------------------------
export_df = high_flagged_df.copy()
export_df["triggered_criteria_list"] = export_df["triggered_criteria"].apply(
    lambda x: "; ".join(x) if isinstance(x, list) else ""
)
export_df["flag_reasons_text"] = export_df["flag_reasons"].apply(
    lambda x: "; ".join(x) if isinstance(x, list) else ""
)

drop_cols = ["triggered_criteria", "flag_reasons"]
export_clean = export_df.drop(columns=[c for c in drop_cols if c in export_df.columns])

csv_data = export_clean.to_csv(index=False).encode("utf-8")

st.download_button(
    label=f"💾 Download High Flagged Transactions CSV ({len(export_clean)} records)",
    data=csv_data,
    file_name=f"high_risk_flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=False,
)

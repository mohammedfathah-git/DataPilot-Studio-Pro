"""
=============================================================================
DataPilot Studio Pro - Enterprise AI & Autonomous Data Analyst Platform
Simplified, Intuitive, Step-by-Step Analytical Experience
=============================================================================
"""
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="DataPilot Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Internal Engine Modules
from modules.data_manager import (
    load_file, clean_name, get_sample_datasets, join_datasets
)
from modules.profiler import (
    profile_dataset, text_columns, numeric_columns, datetime_columns
)
from modules.cleaner import clean_dataset, engineer_features
from modules.statistics import (
    run_ttest, run_anova, run_chisquare, generate_pivot, generate_automated_insights
)
from modules.ml_engine import train_predictor, run_clustering
from modules.visualizer import (
    style_figure, render_chart_with_actions, create_bar_chart,
    create_line_chart, create_scatter_chart, create_distribution_chart,
    create_composition_chart, THEME_PALETTES
)
from modules.exporter import (
    generate_python_script, generate_executive_html_report, export_multi_sheet_excel
)
from modules.ui_theme import inject_modern_theme

# =====================================================================
# SESSION STATE INITIALIZATION & SAFETY GUARDS
# =====================================================================
ss = st.session_state
ss.setdefault("theme_mode", "Dark")
ss.setdefault("step", 1)                  # 1: Upload, 2: Scan, 3: Clean, 4: Charts & Insights
ss.setdefault("datasets", {})             # { table_name: pd.DataFrame }
ss.setdefault("active_dataset_name", "")   # currently active table
ss.setdefault("audit_logs", {})           # { table_name: [log_strings] }
ss.setdefault("chart_palette", "Indigo Modern")

is_dark = ss["theme_mode"] == "Dark"
inject_modern_theme(is_dark)

# Safety Guard: If no data loaded, force Step 1 (Upload)
if not ss["datasets"] or not ss.get("active_dataset_name"):
    ss["step"] = 1

def go_step(step_num: int):
    ss["step"] = step_num

def set_active_dataset(name: str):
    if name in ss["datasets"]:
        ss["active_dataset_name"] = name

def get_current_df() -> pd.DataFrame | None:
    name = ss.get("active_dataset_name")
    if name and name in ss["datasets"]:
        return ss["datasets"][name]
    return None

def update_current_df(new_df: pd.DataFrame, log_msg: str = ""):
    name = ss.get("active_dataset_name")
    if name:
        ss["datasets"][name] = new_df
        if log_msg:
            ss["audit_logs"].setdefault(name, []).append(log_msg)


def inject_modern_theme(is_dark: bool):
    # ... your existing theme variables and CSS ...
    
    # Add this specific CSS override to fix the invisible text
    st.markdown(
        """
        <style>
        /* 1. Fix the File Uploader Text */
        [data-testid="stFileUploader"] span, 
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #F8FAFC !important; /* Forces text to be white/light */
        }
        
        /* 2. Fix the Sidebar "Reset Workspace" Button */
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"] p {
            color: #F8FAFC !important; 
        }
        
        /* 3. Fix the Sidebar Radio Button Labels (Dark / Light) */
        [data-testid="stSidebar"] [role="radiogroup"] label p {
            color: #0F172A !important; /* Force dark text if sidebar is light */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# =====================================================================
# CLEAN, MINIMAL SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <div style="font-size: 22px; background: linear-gradient(135deg,#6366F1,#8B5CF6); border-radius: 9px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; color: white;">⚡</div>
            <div>
                <div style="font-size: 18px; font-weight: 800; letter-spacing: -0.3px; color: {'#F8FAFC' if is_dark else '#0F172A'};">DataPilot <span style="color: #6366F1;">Pro</span></div>
                <div style="font-size: 10px; color: {'#94A3B8' if is_dark else '#64748B'}; font-weight: 600;">YOUR AI DATA ANALYST</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # 4 Simple Steps
    st.caption("WORKFLOW STEPS")
    STEPS = [
        (1, "📤 1. Upload & Combine"),
        (2, "🔍 2. Health & Scan"),
        (3, "✨ 3. Clean & Fix"),
        (4, "📊 4. Charts & Insights"),
    ]

    has_data = bool(ss["datasets"] and ss.get("active_dataset_name"))

    for s_num, label in STEPS:
        is_active = ss["step"] == s_num
        is_disabled = (s_num > 1 and not has_data)
        if st.sidebar.button(
            label,
            key=f"sb_step_{s_num}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_disabled
        ):
            go_step(s_num)
            st.rerun()

    st.divider()

    # Active Table Selector
    if has_data:
        st.caption("CURRENT TABLE")
        ds_keys = list(ss["datasets"].keys())
        active_idx = ds_keys.index(ss["active_dataset_name"]) if ss["active_dataset_name"] in ds_keys else 0
        picked_ds = st.selectbox(
            "Select Table",
            ds_keys,
            index=active_idx,
            label_visibility="collapsed",
            key="sb_pick_table"
        )
        if picked_ds != ss["active_dataset_name"]:
            set_active_dataset(picked_ds)
            st.rerun()

        curr_df = get_current_df()
        if curr_df is not None:
            st.caption(f"📄 **{len(curr_df):,}** rows · **{len(curr_df.columns)}** columns")

        st.divider()

    # Preferences & Reset
    st.caption("SETTINGS")
    theme_choice = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if ss["theme_mode"] == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio"
    )
    if theme_choice != ss["theme_mode"]:
        ss["theme_mode"] = theme_choice
        st.rerun()

    if st.button("🗑 Reset Workspace", use_container_width=True, help="Clear all loaded files and start fresh"):
        ss["datasets"] = {}
        ss["active_dataset_name"] = ""
        ss["audit_logs"] = {}
        ss["step"] = 1
        st.rerun()

# =====================================================================
# TOP STEP PROGRESS BAR
# =====================================================================
step = ss["step"]

step_names = ["1. Upload & Combine", "2. Scan & Health", "3. Clean & Fix", "4. Charts & Insights"]
p_cols = st.columns(4)
for i, col in enumerate(p_cols, start=1):
    with col:
        is_curr = i == step
        is_done = i < step
        bar_bg = "#6366F1" if (is_curr or is_done) else ("#1E293B" if is_dark else "#CBD5E1")
        label_color = "#8B5CF6" if is_curr else ("#10B981" if is_done else ("#64748B" if is_dark else "#475569"))
        st.markdown(
            f"""
            <div style="font-size: 11px; font-weight: 700; color: {label_color}; margin-bottom: 4px;">
                {'✓ ' if is_done else ''}{step_names[i-1]}
            </div>
            <div style="height: 5px; border-radius: 4px; background: {bar_bg}; margin-bottom: 12px;"></div>
            """,
            unsafe_allow_html=True
        )

# =====================================================================
# STEP 1: UPLOAD & MULTI-FILE COMBINE
# =====================================================================
if step == 1:
    st.markdown(
        """
        <div class="dp-hero">
            <div class="dp-hero-icon">📤</div>
            <div>
                <div class="dp-hero-title">Upload Your Data</div>
                <div class="dp-hero-sub">Drop one or more files (CSV, Excel with multiple sheets, JSON). We'll inspect everything automatically.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload your dataset (CSV, Excel, JSON)",
        type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
        key="main_uploader",
        help="You can drag and drop one or several files here."
    )

    if uploaded_files:
        added = 0
        for uf in uploaded_files:
            try:
                tables = load_file(uf.getvalue(), uf.name)
                for t_name, df_table in tables.items():
                    if t_name not in ss["datasets"]:
                        ss["datasets"][t_name] = df_table
                        ss["audit_logs"][t_name] = [f"Loaded '{uf.name}' ({len(df_table):,} rows)."]
                        added += 1
            except Exception as e:
                st.error(f"Could not read {uf.name}: {e}")

        if added > 0:
            if not ss.get("active_dataset_name") and ss["datasets"]:
                set_active_dataset(list(ss["datasets"].keys())[0])
            st.rerun()

    # 1-Click Sample Dataset Options (If nothing loaded yet)
    if not ss["datasets"]:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        st.write("**Or try a ready-to-use sample dataset:**")
        s1, s2, s3 = st.columns(3)
        samples = get_sample_datasets()

        with s1:
            if st.button("🛒 Retail Orders & Customers", use_container_width=True):
                ss["datasets"]["ecommerce_orders"] = samples["ecommerce_orders"]
                ss["datasets"]["ecommerce_customers"] = samples["ecommerce_customers"]
                set_active_dataset("ecommerce_orders")
                st.rerun()
            st.caption("Multi-table dataset with orders, categories, customer IDs, and sales.")

        with s2:
            if st.button("👥 Employee HR Analytics", use_container_width=True):
                ss["datasets"]["employee_hr"] = samples["employee_hr"]
                set_active_dataset("employee_hr")
                st.rerun()
            st.caption("200 employee records with salaries, experience, departments, and attrition.")

        with s3:
            if st.button("🍽️ Restaurant Tips", use_container_width=True):
                ss["datasets"]["restaurant_tips"] = samples["restaurant_tips"]
                set_active_dataset("restaurant_tips")
                st.rerun()
            st.caption("Classic dataset tracking total bill, tips, group size, and meal times.")

    # Show Loaded Tables & Multi-Table Merge
    if ss["datasets"]:
        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        ds_names = list(ss["datasets"].keys())
        
        if len(ds_names) > 1:
            st.markdown(
                f"""
                <div class="dp-card" style="border-left: 4px solid #6366F1;">
                    <div style="font-weight: 700; font-size: 15px;">📁 {len(ds_names)} Tables Loaded in Workspace</div>
                    <div style="font-size: 13px; color: {'#94A3B8' if is_dark else '#475569'}; margin-top: 3px;">
                        Tables: {', '.join([f'<strong>{n}</strong>' for n in ds_names])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Easy Combine / Join Tables Option
            with st.expander("🔗 Combine / Join Two Tables (e.g. Orders + Customers)", expanded=False):
                st.caption("Merge two tables together using a common column (like Customer ID or Order ID).")
                c1, c2 = st.columns(2)
                with c1:
                    t_left = st.selectbox("Table 1", ds_names, index=0, key="join_t1")
                    k_left = st.selectbox("Matching column in Table 1", list(ss["datasets"][t_left].columns), key="join_k1")
                with c2:
                    t_right = st.selectbox("Table 2", ds_names, index=1, key="join_t2")
                    right_cols = list(ss["datasets"][t_right].columns)
                    def_right_idx = right_cols.index(k_left) if k_left in right_cols else 0
                    k_right = st.selectbox("Matching column in Table 2", right_cols, index=def_right_idx, key="join_k2")

                j_type = st.selectbox("Join Type", ["inner", "left", "outer"], index=0, help="'inner' keeps only matching rows, 'left' keeps all rows from Table 1.")
                merged_tbl_name = st.text_input("Name for the combined table", value=f"{t_left}_{t_right}_merged")

                if st.button("⚡ Merge Tables Now", type="primary"):
                    try:
                        merged = join_datasets(
                            ss["datasets"][t_left],
                            ss["datasets"][t_right],
                            left_on=k_left,
                            right_on=k_right,
                            how=j_type
                        )
                        ss["datasets"][merged_tbl_name] = merged
                        ss["audit_logs"][merged_tbl_name] = [f"Merged '{t_left}' and '{t_right}' on {k_left}={k_right}."]
                        set_active_dataset(merged_tbl_name)
                        st.success(f"Merged successfully! Generated '{merged_tbl_name}' with {len(merged):,} rows.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Merge error: {e}")

        # Active Table Preview
        curr_name = ss["active_dataset_name"]
        curr_df = get_current_df()
        
        st.markdown(f"#### Preview: **{curr_name}** ({len(curr_df):,} rows × {len(curr_df.columns)} columns)")
        st.dataframe(curr_df.head(25), use_container_width=True)

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        col_next, _ = st.columns([2.5, 4.5])
        with col_next:
            st.button(
                "Continue to Scan & Health Check →",
                on_click=go_step,
                args=(2,),
                type="primary",
                use_container_width=True
            )

# =====================================================================
# STEP 2: SCAN & DATA HEALTH
# =====================================================================
elif step == 2:
    df = get_current_df()
    curr_name = ss.get("active_dataset_name")

    st.markdown(
        f"""
        <div class="dp-hero">
            <div class="dp-hero-icon">🔍</div>
            <div>
                <div class="dp-hero-title">Data Health Scan: {curr_name}</div>
                <div class="dp-hero-sub">We checked your dataset for missing gaps, duplicate records, formatted numbers, and statistical outliers.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    profile = profile_dataset(df)
    hs = profile["health_score"]
    badge_color = "#10B981" if hs >= 80 else ("#F59E0B" if hs >= 60 else "#EF4444")
    health_title = "High Quality Data" if hs >= 80 else ("Moderate Quality - Minor fixes recommended" if hs >= 60 else "Needs Cleaning")

    # Friendly Health Card
    st.markdown(
        f"""
        <div class="dp-card" style="border-left: 6px solid {badge_color}; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 13px; font-weight: 700; color: {'#94A3B8' if is_dark else '#475569'}; text-transform: uppercase;">Overall Data Health Score</div>
                <div style="font-size: 36px; font-weight: 800; color: {'#F8FAFC' if is_dark else '#0F172A'};">
                    {hs} <span style="font-size: 18px; color: {'#94A3B8' if is_dark else '#64748B'};">/ 100</span>
                </div>
                <div style="font-size: 13.5px; color: {badge_color}; font-weight: 600; margin-top: 2px;">{health_title}</div>
            </div>
            <div>
                <div style="font-size: 12.5px; color: {'#94A3B8' if is_dark else '#475569'}; line-height: 1.6;">
                    <div>• Completeness: <strong>{profile['health_breakdown']['Completeness (35%)']} / 35</strong></div>
                    <div>• Uniqueness: <strong>{profile['health_breakdown']['Uniqueness (25%)']} / 25</strong></div>
                    <div>• Consistency: <strong>{profile['health_breakdown']['Consistency (20%)']} / 20</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Friendly Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{profile['rows']:,}")
    m2.metric("Total Columns", f"{profile['cols']:,}")
    m3.metric("Missing Values", f"{profile['missing_cells']:,}", f"{profile['missing_pct']}% of data", delta_color="inverse")
    m4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}", f"{profile['duplicate_pct']}% duplicates", delta_color="inverse")

    # Plain-English Issue Summary
    issues = []
    if profile["missing_cells"]:
        issues.append(f"Found **{profile['missing_cells']:,}** empty cells that should be filled.")
    if profile["duplicate_rows"]:
        issues.append(f"Found **{profile['duplicate_rows']:,}** duplicate rows that can be removed.")
    if profile["blank_cells"] or profile["spacey_cells"]:
        issues.append(f"Found **{profile['blank_cells'] + profile['spacey_cells']:,}** text cells with extra spaces or empty text.")
    if profile["numeric_like"]:
        issues.append(f"Numbers saved as text: {', '.join([f'**{c}**' for c in profile['numeric_like']])}.")
    if profile["outlier_rows_count"]:
        issues.append(f"Found **{profile['outlier_rows_count']:,}** rows with extreme outlier values.")

    if issues:
        st.markdown(
            f"""
            <div class="dp-card">
                <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px;">📋 Things We Found:</div>
                <div style="font-size: 13.5px; line-height: 1.7;">
                    {''.join([f'<div>• {iss}</div>' for iss in issues])}
                </div>
                <div style="font-size: 12px; color: {'#94A3B8' if is_dark else '#475569'}; margin-top: 10px;">
                    💡 You can fix all of these with 1 click in the next step!
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success("🎉 Outstanding! Your dataset has zero missing values and zero duplicate rows.")

    # Column Summary Table
    with st.expander("👀 View Details for Every Column", expanded=False):
        st.dataframe(profile["columns_summary"][["Column", "Semantic Type", "Missing", "Missing (%)", "Unique", "Mean", "Median", "Top Value"]], use_container_width=True)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    b_back, b_next, _ = st.columns([1.5, 2.5, 4])
    with b_back:
        st.button("← Back to Upload", on_click=go_step, args=(1,), use_container_width=True)
    with b_next:
        st.button("Continue to Clean & Fix →", on_click=go_step, args=(3,), type="primary", use_container_width=True)

# =====================================================================
# STEP 3: CLEAN & FIX
# =====================================================================
elif step == 3:
    df = get_current_df()
    curr_name = ss.get("active_dataset_name")

    st.markdown(
        f"""
        <div class="dp-hero">
            <div class="dp-hero-icon">✨</div>
            <div>
                <div class="dp-hero-title">Clean & Fix Your Data</div>
                <div class="dp-hero-sub">Fix missing values, strip spaces, convert currency symbols to numbers, and cap extreme outliers.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    profile = profile_dataset(df)

    # Magic 1-Click Clean
    st.markdown(
        f"""
        <div class="dp-card" style="border: 2px solid #6366F1; background: {'rgba(99,102,241,0.06)' if is_dark else '#EEF2FF'};">
            <div style="font-size: 18px; font-weight: 800; color: {'#818CF8' if is_dark else '#4F46E5'};">✨ Magic 1-Click Clean</div>
            <div style="font-size: 13.5px; color: {'#94A3B8' if is_dark else '#475569'}; margin-top: 4px;">
                Recommended for everyone. Automatically removes duplicate rows, trims messy whitespace, converts currency/percentages to numbers, fills missing gaps with medians, and caps extreme outliers.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚀 Run Magic 1-Click Clean", type="primary", use_container_width=True):
        preset_opts = {
            "trim": True,
            "text_case": "Title Case",
            "to_numeric": profile["numeric_like"],
            "dedupe": True,
            "drop_constant": True,
            "col_thr": 85,
            "num_strategy": "Fill with median",
            "txt_strategy": "Fill with mode (most frequent)",
            "outlier_action": "Winsorize (Cap at 1st & 99th percentiles)"
        }
        with st.spinner("Cleaning and standardizing your dataset..."):
            cleaned_df, log_lines = clean_dataset(df, preset_opts)
            for line in log_lines:
                update_current_df(cleaned_df, line)
        st.success("Your dataset is now sparkling clean!")
        st.rerun()

    # Optional Custom Adjustments Expander
    with st.expander("⚙️ Advanced Custom Adjustments (Optional)", expanded=False):
        st.caption("Customize specific cleaning rules if you have specific business requirements.")
        c1, c2 = st.columns(2)
        with c1:
            opt_dedupe = st.checkbox("Remove Duplicate Rows", value=bool(profile["duplicate_rows"]))
            opt_trim = st.checkbox("Trim Extra Spaces & Blanks", value=True)
            opt_num_strat = st.selectbox("How to fill missing numbers?", ["Fill with median", "Fill with mean", "Fill with 0", "Leave as is", "Drop those rows"])
        with c2:
            opt_txt_strat = st.selectbox("How to fill missing text?", ["Fill with mode (most frequent)", "Fill with 'Unknown'", "Leave as is", "Drop those rows"])
            opt_outliers = st.selectbox("Outliers", ["Winsorize (Cap at 1st & 99th percentiles)", "Drop outlier rows", "Leave as is"])

        if st.button("Apply Custom Rules"):
            custom_opts = {
                "dedupe": opt_dedupe,
                "trim": opt_trim,
                "text_case": "None",
                "to_numeric": profile["numeric_like"],
                "num_strategy": opt_num_strat,
                "txt_strategy": opt_txt_strat,
                "outlier_action": opt_outliers,
                "col_thr": 100
            }
            with st.spinner("Applying custom rules..."):
                cleaned_df, log_lines = clean_dataset(df, custom_opts)
                for line in log_lines:
                    update_current_df(cleaned_df, line)
            st.success("Custom rules applied!")
            st.rerun()

    # Applied Cleaning Log
    audit_log = ss["audit_logs"].get(curr_name, [])
    if len(audit_log) > 1:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        st.markdown("##### 📜 What Was Cleaned:")
        for line in audit_log[1:]:
            st.markdown(f"• {line}")

    # Clean Preview
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    st.markdown(f"#### Cleaned Dataset Preview ({len(df):,} rows)")
    st.dataframe(df.head(25), use_container_width=True)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    b_back, b_next, _ = st.columns([1.5, 2.5, 4])
    with b_back:
        st.button("← Back to Scan", on_click=go_step, args=(2,), use_container_width=True)
    with b_next:
        st.button("Continue to Charts & Insights →", on_click=go_step, args=(4,), type="primary", use_container_width=True)

# =====================================================================
# STEP 4: CHARTS & INSIGHTS (STREAMLINED & UNIFIED)
# =====================================================================
elif step == 4:
    df = get_current_df()
    curr_name = ss.get("active_dataset_name")

    st.markdown(
        f"""
        <div class="dp-hero">
            <div class="dp-hero-icon">📊</div>
            <div>
                <div class="dp-hero-title">Visual Charts & AI Insights</div>
                <div class="dp-hero-sub">Explore your data with live data labels, download reports, or inspect automated key business takeaways.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    all_cols = list(df.columns)
    nums = numeric_columns(df)
    cats = text_columns(df)

    # 1. Easy Chart Controls
    st.markdown("#### Easy Chart Builder")
    
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        c_kind = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie / Donut Chart", "Histogram"])
    with ctrl_col2:
        x_col = st.selectbox("Category (X-Axis)", all_cols, index=0)
    with ctrl_col3:
        # Default to a numeric column if available
        def_y_idx = 0 if nums else 0
        y_col = st.selectbox("Value Metric (Y-Axis)", nums if nums else all_cols, index=def_y_idx)
    with ctrl_col4:
        color_by = st.selectbox("Color / Breakdown", ["None"] + cats, index=0)

    # Chart Options (Data Labels + Aggregation)
    opt_col1, opt_col2, opt_col3 = st.columns([2.5, 2.5, 3])
    with opt_col1:
        # User requested: turn on chart data label by default!
        show_labels = st.checkbox("🏷️ Show Data Labels on Chart", value=True)
    with opt_col2:
        # Smart aggregation: Average or Sum
        is_avg_like = any(k in str(y_col).lower() for k in ["avg", "rate", "pct", "percent", "ratio", "salary", "score"])
        agg_mode = st.selectbox("Calculation", ["Average (Mean)", "Total (Sum)", "Count"], index=0 if is_avg_like else 1)
    with opt_col3:
        sort_desc = st.checkbox("Sort highest to lowest", value=True)

    # Render Chart
    color_arg = None if color_by == "None" else color_by
    palette = ss.get("chart_palette", "Indigo Modern")

    if c_kind == "Bar Chart":
        # Group and aggregate for high visual quality
        group_keys = [x_col] + ([color_by] if color_arg else [])
        if agg_mode == "Average (Mean)":
            plot_df = df.groupby(group_keys)[y_col].mean().round(2).reset_index()
            chart_title = f"Average {y_col} by {x_col}"
        elif agg_mode == "Count":
            plot_df = df.groupby(group_keys).size().reset_index(name="Count")
            y_col_plot = "Count"
            chart_title = f"Count of Records by {x_col}"
        else:
            plot_df = df.groupby(group_keys)[y_col].sum().round(2).reset_index()
            chart_title = f"Total {y_col} by {x_col}"
            
        y_actual = "Count" if agg_mode == "Count" else y_col
        if sort_desc and not color_arg:
            plot_df = plot_df.sort_values(y_actual, ascending=False)
            
        fig = create_bar_chart(
            plot_df.head(25),
            x=x_col,
            y=y_actual,
            color=color_arg,
            palette=palette,
            show_labels=show_labels
        )
        styled_fig = render_chart_with_actions(fig, key="unified_chart", is_dark=is_dark, title=chart_title)

    elif c_kind == "Line Chart":
        agg_line = df.groupby(x_col)[y_col].mean().reset_index() if agg_mode == "Average (Mean)" else df.groupby(x_col)[y_col].sum().reset_index()
        fig = create_line_chart(agg_line, x=x_col, y=y_col, palette=palette, show_labels=show_labels)
        styled_fig = render_chart_with_actions(fig, key="unified_chart", is_dark=is_dark, title=f"{y_col} over {x_col}")

    elif c_kind == "Scatter Plot":
        fig = create_scatter_chart(df, x=x_col, y=y_col, color=color_arg, trendline=True, palette=palette, show_labels=show_labels)
        styled_fig = render_chart_with_actions(fig, key="unified_chart", is_dark=is_dark, title=f"{y_col} vs {x_col}")

    elif c_kind == "Pie / Donut Chart":
        agg_pie = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(10)
        fig = create_composition_chart(agg_pie, names=x_col, values=y_col, kind="Donut", palette=palette, show_labels=show_labels)
        styled_fig = render_chart_with_actions(fig, key="unified_chart", is_dark=is_dark, title=f"Share of {y_col} by {x_col}")

    elif c_kind == "Histogram":
        fig = create_distribution_chart(df, col=y_col, kind="Histogram", palette=palette, show_labels=show_labels)
        styled_fig = render_chart_with_actions(fig, key="unified_chart", is_dark=is_dark, title=f"Distribution of {y_col}")

    # 2. Quick Download Bar (Directly below chart - No extra tabs needed!)
    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    st.markdown("##### 💾 Quick Downloads")
    
    dl_c1, dl_c2, dl_c3, dl_c4 = st.columns(4)
    profile = profile_dataset(df)
    audit_log = ss["audit_logs"].get(curr_name, [])

    with dl_c1:
        st.download_button(
            "⬇ Cleaned CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{curr_name}_cleaned.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
    with dl_c2:
        excel_bytes = export_multi_sheet_excel(df, profiler_summary=profile["columns_summary"])
        st.download_button(
            "⬇ Excel (.xlsx)",
            excel_bytes,
            file_name=f"{curr_name}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with dl_c3:
        html_chart_bytes = styled_fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "⬇ Chart (HTML)",
            html_chart_bytes,
            file_name=f"{curr_name}_chart.html",
            mime="text/html",
            use_container_width=True
        )
    with dl_c4:
        html_report = generate_executive_html_report(
            dataset_name=curr_name,
            health_score=profile["health_score"],
            health_breakdown=profile["health_breakdown"],
            n_rows=len(df),
            n_cols=len(df.columns),
            clean_log=audit_log,
            insights=generate_automated_insights(df)
        )
        st.download_button(
            "⬇ Executive Brief",
            html_report.encode("utf-8"),
            file_name=f"{curr_name}_brief.html",
            mime="text/html",
            use_container_width=True
        )

    # 3. AI Automated Key Insights (Clear, simple card right below)
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    st.markdown("##### 💡 Key Automated Insights")
    insights = generate_automated_insights(df)
    
    ins_cols = st.columns(min(len(insights), 3))
    for i, item in enumerate(insights[:3]):
        with ins_cols[i]:
            st.markdown(
                f"""
                <div class="dp-card" style="border-left: 4px solid #6366F1; padding: 14px 18px;">
                    <div style="font-weight: 700; font-size: 14px;">{item['icon']} {item['title']}</div>
                    <div style="font-size: 12.5px; color: {'#94A3B8' if is_dark else '#475569'}; margin-top: 4px;">{item['text']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Optional: Predictor Expander for deeper dive
    with st.expander("🎯 What Drives This Number? (Analyze Key Drivers)", expanded=False):
        st.caption("Pick a target metric. Our AI will rank which factors impact it the most.")
        p1, p2 = st.columns(2)
        with p1:
            p_target = st.selectbox("Target to analyze", all_cols, index=len(all_cols)-1, key="exp_target_pred")
        with p2:
            p_cand = [c for c in all_cols if c != p_target]
            p_features = st.multiselect("Predictor factors", p_cand, default=p_cand[:6], key="exp_features_pred")

        
if st.button("🚀 Analyze Drivers Now", type="primary"):
    with st.spinner("Training and comparing ML models..."):
        try:
            ml_res = train_predictor(df, p_target, p_features)

            # 1. Display selected model
            st.success(
                f"Best Model: {ml_res['best_model_name']}"
            )

            # 2. Model comparison dashboard
            st.markdown("### 🏆 Model Performance Comparison")

            comparison_df = pd.DataFrame(
                ml_res["model_comparison"]
            ).T

            st.dataframe(
                comparison_df,
                use_container_width=True
            )

            # 3. Display training information
            st.caption(
                f"Training samples: {ml_res['train_samples']:,} | "
                f"Testing samples: {ml_res['test_samples']:,}"
            )

            # 4. Feature importance chart
            st.markdown("### 📊 Feature Importance")

            fi_fig = px.bar(
                ml_res["feature_importance"],
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Purples",
                title=f"What impacts '{p_target}' the most?"
            )

            render_chart_with_actions(
                fi_fig,
                key="step4_driver_chart",
                is_dark=is_dark
            )

        except Exception as e:
            st.error(f"Could not calculate drivers: {e}")


    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    st.button("← Back to Clean & Fix", on_click=go_step, args=(3,))

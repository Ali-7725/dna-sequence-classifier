"""
DNA Sequence Classifier - Streamlit Web Application
Computational Biology | Assignment 2
High-Performance Vercel-Style Developer Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_extractor import (
    sanitize_sequence,
    validate_dna_sequence,
    extract_features,
    extract_batch_features,
)
from ml_engine import (
    DNAClassifierPipeline,
    get_training_dataframe,
    ASSIGNMENT_CORE_DATASET,
    FEATURE_COLUMNS,
)
from academic_content import (
    ASSIGNMENT_FEATURE_TABLE_MD,
    ACADEMIC_SECTIONS,
    generate_markdown_report,
)


# --- Page Configuration ---
st.set_page_config(
    page_title="BioSeq-AI | DNA Sequence Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Vercel-Inspired Sleek Dark Theme Custom CSS ---
CUSTOM_CSS = """
<style>
/* Global styles */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Containers */
.main {
    background-color: #090d16;
}

/* Card Container */
.custom-card {
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.custom-card-glow {
    background: #0f172a;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.12);
}

/* Pill Badges */
.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);
    color: #06b6d4;
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}

.badge-purple {
    background: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}

.seq-code {
    font-family: 'JetBrains Mono', monospace;
    background: #1e293b;
    color: #38bdf8;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
}

/* Prediction Box Hero */
.prediction-hero-b {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.12) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}

.prediction-hero-a {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(236, 72, 153, 0.12) 100%);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}

/* Stat Metric Block */
.stat-box {
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.stat-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}
.stat-lbl {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Session State Initialization ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = DNAClassifierPipeline(model_type="Random Forest", n_estimators=50, max_depth=4)
    # Train default core dataset
    default_df = get_training_dataframe("core")
    st.session_state.train_df = default_df
    st.session_state.metrics = st.session_state.pipeline.train(default_df)

if "dataset_mode" not in st.session_state:
    st.session_state.dataset_mode = "core"

if "input_sequence" not in st.session_state:
    st.session_state.input_sequence = "ATGCGCGATC"


# --- Top Header Section ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <span style="font-size: 2.2rem;">🧬</span>
        <div>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em;">BioSeq-AI <span style="font-weight: 300; color: #94a3b8;">| DNA Sequence Classifier</span></h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.88rem;">Assignment 2 &bull; Introduction to Computational Biology &bull; Supervised Machine Learning Pipeline</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge-emerald">&bull; Model Live</span>
        <span class="badge-cyan">Scikit-Learn</span>
        <span class="badge-purple">BioPython-Ready</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0 24px 0;'/>", unsafe_allow_html=True)


# --- Sidebar Navigation & Model Tuning ---
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    
    st.markdown("#### 1. Training Dataset")
    ds_option = st.radio(
        "Select Dataset:",
        options=["Assignment Core (S1-S4)", "Expanded Biological Dataset (24 Sequences)"],
        index=0 if st.session_state.dataset_mode == "core" else 1,
        help="Core: exactly S1-S4 as described in assignment. Expanded: includes 20 additional biological CpG and AT sequences."
    )
    new_mode = "core" if "Core" in ds_option else "expanded"

    st.markdown("#### 2. Model Architecture")
    model_choice = st.selectbox(
        "Classifier Engine:",
        ["Random Forest", "Decision Tree", "Logistic Regression", "Support Vector Machine"],
        index=0
    )

    col_t1, col_t2 = st.columns(2)
    if model_choice in ["Random Forest", "Decision Tree"]:
        with col_t1:
            max_depth = st.slider("Max Depth", min_value=1, max_value=8, value=3)
        with col_t2:
            n_trees = st.slider("Trees (RF)", min_value=10, max_value=100, value=50, step=10) if model_choice == "Random Forest" else 1
    else:
        max_depth = 3
        n_trees = 50

    if st.button("⚡ Retrain Model", use_container_width=True, type="primary"):
        st.session_state.dataset_mode = new_mode
        st.session_state.train_df = get_training_dataframe(new_mode)
        st.session_state.pipeline = DNAClassifierPipeline(
            model_type=model_choice,
            max_depth=max_depth,
            n_estimators=n_trees
        )
        st.session_state.metrics = st.session_state.pipeline.train(st.session_state.train_df)
        st.success(f"Model retrained: {model_choice} ({new_mode.capitalize()} Dataset)")

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'/>", unsafe_allow_html=True)

    st.markdown("#### 📌 Assignment Sequences")
    st.markdown("""
    - **S1:** `ATGCGATC` *(Group A)*
    - **S2:** `ATGCGTAC` *(Group A)*
    - **S3:** `GCGCGTAT` *(Group B)*
    - **S4:** `GCGCGCGC` *(Group B)*
    - **Test:** `ATGCGCGATC` *(Unknown)*
    """)

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'/>", unsafe_allow_html=True)
    st.caption("Computational Biology DNA Analytics • 2026")


# --- Main Application Tabs ---
tab_predict, tab_dataset, tab_ml_metrics, tab_academic = st.tabs([
    "🔮 Sequence Classifier & Predictor",
    "📊 Training Dataset & Feature Engine",
    "🤖 ML Pipeline & Model Insights",
    "📝 Academic Assignment Submission",
])


# ==============================================================================
# TAB 1: INTERACTIVE SEQUENCE CLASSIFIER & PREDICTOR
# ==============================================================================
with tab_predict:
    st.markdown("### 🧪 Interactive DNA Sequence Classification")
    st.markdown("Enter any custom DNA sequence to extract biophysical features and predict its group membership (`Group A` vs `Group B`).")

    # Preset Quick Buttons
    st.markdown("<span style='font-size: 0.85rem; color: #94a3b8;'>Preset Sequence Presets:</span>", unsafe_allow_html=True)
    p_cols = st.columns(5)
    if p_cols[0].button("Assignment Test (`ATGCGCGATC`)", use_container_width=True):
        st.session_state.input_sequence = "ATGCGCGATC"
    if p_cols[1].button("Canonical S1 (`ATGCGATC`)", use_container_width=True):
        st.session_state.input_sequence = "ATGCGATC"
    if p_cols[2].button("Canonical S4 (`GCGCGCGC`)", use_container_width=True):
        st.session_state.input_sequence = "GCGCGCGC"
    if p_cols[3].button("High GC (90% GC)", use_container_width=True):
        st.session_state.input_sequence = "CGCGCCGCGG"
    if p_cols[4].button("AT-Rich (20% GC)", use_container_width=True):
        st.session_state.input_sequence = "AATATATATA"

    user_seq = st.text_area(
        "Input DNA Sequence (Canonical A, T, G, C):",
        value=st.session_state.input_sequence,
        height=75,
        help="Paste a raw sequence or FASTA formatted sequence. Whitespace and non-letters are automatically cleaned."
    )

    clean_user_seq = sanitize_sequence(user_seq)
    is_valid, val_msg, invalid_chars = validate_dna_sequence(clean_user_seq)

    if not clean_user_seq:
        st.warning("Please input a DNA sequence above to analyze.")
    elif not is_valid:
        st.error(f"❌ Validation Error: {val_msg}")
    else:
        # Run Prediction
        prediction_result = st.session_state.pipeline.predict_sequence(clean_user_seq, seq_id="User_Input")
        feat = prediction_result["features"]
        pred_class = prediction_result["predicted_class"]
        confidence = prediction_result["confidence_pct"]
        prob_a = prediction_result["prob_group_a_pct"]
        prob_b = prediction_result["prob_group_b_pct"]

        # Hero Prediction Result Banner
        hero_class = "prediction-hero-b" if pred_class == "Group B" else "prediction-hero-a"
        badge_style = "badge-emerald" if pred_class == "Group B" else "badge-purple"
        pred_desc = "GC-Enriched Sequence Motif" if pred_class == "Group B" else "Balanced / AT-Moderate Motif"

        st.markdown(f"""
        <div class="{hero_class}">
            <span class="{badge_style}" style="font-size: 0.9rem; margin-bottom: 8px;">Prediction Result</span>
            <h2 style="margin: 8px 0 4px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;">
                {pred_class}
            </h2>
            <p style="margin: 0; color: #cbd5e1; font-size: 1rem;">
                Confidence: <strong style="color: #f8fafc; font-size: 1.15rem;">{confidence}%</strong> &bull; {pred_desc}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Distribution Bar
        st.markdown("<p style='font-size: 0.82rem; color: #94a3b8; margin: 12px 0 4px 0;'>Model Probability Distribution:</p>", unsafe_allow_html=True)
        prob_col1, prob_col2 = st.columns([1, 1])
        with prob_col1:
            st.write(f"**Group A:** {prob_a}%")
            st.progress(prob_a / 100.0)
        with prob_col2:
            st.write(f"**Group B:** {prob_b}%")
            st.progress(prob_b / 100.0)

        # Quick Metric Cards
        st.markdown("<h4 style='margin-top: 24px; margin-bottom: 12px;'>🧬 Computed Biophysical Features</h4>", unsafe_allow_html=True)
        m_cols = st.columns(6)
        with m_cols[0]:
            st.markdown(f"""<div class="stat-box"><div class="stat-val">{feat['Length']}</div><div class="stat-lbl">Length (bp)</div></div>""", unsafe_allow_html=True)
        with m_cols[1]:
            st.markdown(f"""<div class="stat-box"><div class="stat-val" style="color: #10b981;">{feat['GC_Content_%']}%</div><div class="stat-lbl">GC Content</div></div>""", unsafe_allow_html=True)
        with m_cols[2]:
            st.markdown(f"""<div class="stat-box"><div class="stat-val" style="color: #38bdf8;">{feat['Count_A']}</div><div class="stat-lbl">Adenine (A)</div></div>""", unsafe_allow_html=True)
        with m_cols[3]:
            st.markdown(f"""<div class="stat-box"><div class="stat-val" style="color: #f43f5e;">{feat['Count_T']}</div><div class="stat-lbl">Thymine (T)</div></div>""", unsafe_allow_html=True)
        with m_cols[4]:
            st.markdown(f"""<div class="stat-box"><div class="stat-val" style="color: #10b981;">{feat['Count_G']}</div><div class="stat-lbl">Guanine (G)</div></div>""", unsafe_allow_html=True)
        with m_cols[5]:
            st.markdown(f"""<div class="stat-box"><div class="stat-val" style="color: #fbbf24;">{feat['Count_C']}</div><div class="stat-lbl">Cytosine (C)</div></div>""", unsafe_allow_html=True)

        # Detailed Breakdown & Visualizations
        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            # Nucleotide Bar Chart
            base_df = pd.DataFrame({
                "Nucleotide": ["Adenine (A)", "Thymine (T)", "Guanine (G)", "Cytosine (C)"],
                "Count": [feat["Count_A"], feat["Count_T"], feat["Count_G"], feat["Count_C"]],
                "Color": ["#38bdf8", "#f43f5e", "#10b981", "#fbbf24"]
            })
            fig_bases = px.bar(
                base_df,
                x="Nucleotide",
                y="Count",
                color="Nucleotide",
                color_discrete_map={
                    "Adenine (A)": "#38bdf8",
                    "Thymine (T)": "#f43f5e",
                    "Guanine (G)": "#10b981",
                    "Cytosine (C)": "#fbbf24"
                },
                title="Nucleotide Composition Breakdown",
                text="Count"
            )
            fig_bases.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#f8fafc",
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                height=320,
            )
            st.plotly_chart(fig_bases, use_container_width=True)

        with viz_col2:
            # GC Content Indicator Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=feat["GC_Content_%"],
                title={"text": "GC Content Percentage (%)", "font": {"size": 16, "color": "#f8fafc"}},
                number={"suffix": "%", "font": {"size": 28, "color": "#10b981"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#10b981"},
                    "bgcolor": "#1e293b",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 55], "color": "rgba(139, 92, 246, 0.2)"},
                        {"range": [55, 100], "color": "rgba(16, 185, 129, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f43f5e", "width": 3},
                        "thickness": 0.75,
                        "value": 56.25  # Theoretical decision threshold
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0f172a",
                font_color="#f8fafc",
                margin=dict(l=25, r=25, t=40, b=25),
                height=320,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Extended Biological Features Table
        with st.expander("🔬 View Extended Genomic Features (CpG, Purines/Pyrimidines, Skew)"):
            ext_data = {
                "Feature": [
                    "Sequence Cleaned",
                    "Total Length (bp)",
                    "Adenine (A) Count",
                    "Thymine (T) Count",
                    "Guanine (G) Count",
                    "Cytosine (C) Count",
                    "GC Content (%)",
                    "AT Content (%)",
                    "GC / AT Ratio",
                    "Total Purines (A + G)",
                    "Total Pyrimidines (C + T)",
                    "Purine / Pyrimidine Ratio",
                    "CpG Dinucleotide Count",
                    "GC Skew [(G - C) / (G + C)]"
                ],
                "Value": [
                    feat["Sequence"],
                    feat["Length"],
                    feat["Count_A"],
                    feat["Count_T"],
                    feat["Count_G"],
                    feat["Count_C"],
                    f"{feat['GC_Content_%']}%",
                    f"{feat['AT_Content_%']}%",
                    feat["GC_AT_Ratio"],
                    feat["Purines (A+G)"],
                    feat["Pyrimidines (C+T)"],
                    feat["Purine/Pyrimidine_Ratio"],
                    feat["CpG_Count"],
                    feat["GC_Skew"]
                ],
                "Biological Interpretation": [
                    "Sanitized 5' to 3' single-strand primary DNA sequence",
                    "Polymer chain length in base pairs",
                    "Purine nucleotide count; pairs with Thymine via 2 hydrogen bonds",
                    "Pyrimidine nucleotide count; pairs with Adenine via 2 hydrogen bonds",
                    "Purine nucleotide count; pairs with Cytosine via 3 hydrogen bonds",
                    "Pyrimidine nucleotide count; pairs with Guanine via 3 hydrogen bonds",
                    "Key indicator of thermodynamic stability and melting temperature (Tm)",
                    "Complement to GC content; characterizes AT-rich promoter elements (TATA box)",
                    "Biophysical ratio governing double helix thermodynamic resistance",
                    "Two-ring nitrogenous heterocyclic bases",
                    "Single-ring nitrogenous heterocyclic bases",
                    "Chargaff symmetry proxy across chromosomal strands",
                    "Target for vertebrate DNA methyltransferases (DNMTs) and epigenetic regulation",
                    "Strand asymmetry metric often indicating leading vs lagging replication strands"
                ]
            }
            st.dataframe(pd.DataFrame(ext_data), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: TRAINING DATASET & FEATURE EXTRACTION ENGINE
# ==============================================================================
with tab_dataset:
    st.markdown("### 📊 Training Dataset & Feature Matrix")
    st.markdown("Inspect the feature-engineered matrix used to train the machine learning classifier.")

    train_df = st.session_state.train_df
    st.dataframe(
        train_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sequence": st.column_config.TextColumn("DNA Sequence", help="5' -> 3' primary sequence"),
            "GC_Content_%": st.column_config.ProgressColumn(
                "GC Content (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
        }
    )

    # Feature Distribution Plots
    st.markdown("#### 📈 Distribution Analysis across Groups")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig_box = px.box(
            train_df,
            x="Label",
            y="GC_Content_%",
            color="Label",
            color_discrete_map={"Group A": "#a78bfa", "Group B": "#10b981"},
            points="all",
            title="GC Content (%) Separation between Group A & Group B",
        )
        fig_box.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#f8fafc",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d2:
        fig_scatter = px.scatter(
            train_df,
            x="Count_G",
            y="Count_C",
            size="GC_Content_%",
            color="Label",
            color_discrete_map={"Group A": "#a78bfa", "Group B": "#10b981"},
            hover_data=["ID", "Sequence", "GC_Content_%"],
            title="Guanine (G) vs Cytosine (C) Counts",
        )
        fig_scatter.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#f8fafc",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature Extraction Sandbox
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;'/>", unsafe_allow_html=True)
    st.markdown("#### 🧪 Standalone Batch Feature Extraction Sandbox")
    batch_input = st.text_area(
        "Paste Multi-Sequence Input (one per line, format: ID,SEQUENCE or FASTA):",
        value="Seq_Alpha,ATGCGATC\nSeq_Beta,GCGCGCGC\nSeq_Gamma,TATATATA\nSeq_Delta,CGCGCGTA",
        height=100
    )
    if st.button("Process Batch Sequences"):
        lines = [line.strip() for line in batch_input.strip().splitlines() if line.strip()]
        tuples = []
        for idx, line in enumerate(lines):
            if "," in line:
                sid, sseq = line.split(",", 1)
            else:
                sid = f"Seq_{idx+1}"
                sseq = line
            tuples.append((sid.strip(), sseq.strip(), "Unknown"))
        batch_df = extract_batch_features(tuples, include_extended=True)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        csv_data = batch_df.to_csv(index=False)
        st.download_button("📥 Download Extracted Features CSV", csv_data, "batch_dna_features.csv", "text/csv")


# ==============================================================================
# TAB 3: ML PIPELINE & MODEL INSIGHTS
# ==============================================================================
with tab_ml_metrics:
    st.markdown("### 🤖 Machine Learning Model Evaluation & Interpretability")
    metrics = st.session_state.metrics

    # Top Metric Scorecards
    score_cols = st.columns(5)
    with score_cols[0]:
        st.metric("Training Accuracy", f"{metrics['accuracy']}%")
    with score_cols[1]:
        loo = metrics.get('loo_cv_accuracy')
        st.metric("LOOCV Accuracy", f"{loo}%" if loo is not None else "N/A")
    with score_cols[2]:
        st.metric("Precision (Group B)", f"{metrics['precision']}%")
    with score_cols[3]:
        st.metric("Recall (Group B)", f"{metrics['recall']}%")
    with score_cols[4]:
        st.metric("F1-Score", f"{metrics['f1_score']}%")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        # Confusion Matrix Heatmap
        cm = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Group A", "Predicted Group B"],
            y=["Actual Group A", "Actual Group B"],
            color_continuous_scale="Viridis",
            title="Confusion Matrix Heatmap",
        )
        fig_cm.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#f8fafc",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        # Feature Importance Plot
        feat_imp = metrics["feature_importances"]
        if feat_imp:
            imp_df = pd.DataFrame({
                "Feature": list(feat_imp.keys()),
                "Importance": list(feat_imp.values())
            }).sort_values(by="Importance", ascending=True)

            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Scikit-Learn Feature Importance Ranking",
                color="Importance",
                color_continuous_scale="Teal",
            )
            fig_imp.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#f8fafc",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is not directly computed for this model type.")

    # Decision Logic Inspector
    if metrics.get("tree_rules"):
        with st.expander("🌲 Decision Tree Split Logic (Human-Readable Rules)"):
            st.markdown("Below is the exact decision boundary tree synthesized by Scikit-Learn:")
            st.code(metrics["tree_rules"], language="text")


# ==============================================================================
# TAB 4: ACADEMIC ASSIGNMENT SUBMISSION REPORT
# ==============================================================================
with tab_academic:
    st.markdown("### 📝 Assignment 2 Formal Academic Submission Report")
    st.markdown("Comprehensive analysis formatted to directly satisfy all assignment questions for **Introduction to Computational Biology**.")

    # Always compute the prediction for the standard test sequence ATGCGCGATC
    standard_test_pred = st.session_state.pipeline.predict_sequence("ATGCGCGATC", seq_id="Assignment_Test_Sequence")
    markdown_report = generate_markdown_report(st.session_state.metrics, standard_test_pred)

    # Action Buttons: Download
    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        st.download_button(
            "📥 Download Complete Report (Markdown .md)",
            data=markdown_report,
            file_name="assignment2_computational_biology_report.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary"
        )
    with d_col2:
        # Prepare feature table CSV
        core_and_test_tuples = ASSIGNMENT_CORE_DATASET + [("Test", "ATGCGCGATC", "Group B (Predicted)")]
        core_test_df = extract_batch_features(core_and_test_tuples, include_extended=False)
        st.download_button(
            "📊 Download Assignment Feature Table (CSV)",
            data=core_test_df.to_csv(index=False),
            file_name="assignment2_feature_table.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'/>", unsafe_allow_html=True)

    # Section 1: Structured Feature Table
    st.markdown("#### Question 1: Feature Extraction Table")
    st.markdown(ASSIGNMENT_FEATURE_TABLE_MD)

    # Section 2: Model Performance & Evaluation
    st.markdown(ACADEMIC_SECTIONS["model_evaluation"]["title"])
    st.markdown(ACADEMIC_SECTIONS["model_evaluation"]["content"])

    # Section 3: Feature Analysis & Biological Relevance
    st.markdown(ACADEMIC_SECTIONS["feature_analysis"]["title"])
    st.markdown(ACADEMIC_SECTIONS["feature_analysis"]["content"])

    # Section 4: Real-World Genomics Application
    st.markdown(ACADEMIC_SECTIONS["real_world_application"]["title"])
    st.markdown(ACADEMIC_SECTIONS["real_world_application"]["content"])

    # Section 5: Critical Limitations of AI/ML in Genomics
    st.markdown(ACADEMIC_SECTIONS["limitations"]["title"])
    st.markdown(ACADEMIC_SECTIONS["limitations"]["content"])

from __future__ import annotations

import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(
    page_title="ECG Rhythm Intelligence Studio",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------- Theme / styling -------------------------------
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid #e5eaf0;}
.hero {padding: 1.45rem 1.7rem; border-radius: 18px; color:#fff;
background:linear-gradient(115deg,#102b46 0%,#155b70 55%,#258b83 100%); margin-bottom:1rem;}
.hero h1 {color:#fff; font-size:2.05rem; margin:0 0 .3rem 0;}
.hero p {color:#d8edf0; margin:0; font-size:1rem;}
.eyebrow {color:#9fe0d5; font-size:.72rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase;}
.notice {background:#fff8e9; border-left:4px solid #dca33c; color:#594a2a;
padding:.85rem 1rem; border-radius:8px; font-size:.9rem;}
div[data-testid="stMetric"] {background:white; border:1px solid #e7edf2; border-radius:13px; padding:.75rem 1rem;}
.small {color:#758399; font-size:.82rem;}
</style>
""", unsafe_allow_html=True)

RHYTHM_MAP = {
    "SB": "SB",
    "AFIB": "AFIB", "AF": "AFIB",
    "ST": "GSVT", "SVT": "GSVT", "AT": "GSVT",
    "AVNRT": "GSVT", "AVRT": "GSVT", "SAAWR": "GSVT",
    "SR": "SR", "SI": "SR", "SA": "SR"
}
GROUPS = ["SB", "AFIB", "GSVT", "SR"]
GROUP_NAMES = {
    "SB": "Sinus Bradycardia",
    "AFIB": "Atrial Fibrillation/Flutter",
    "GSVT": "Generalized Supraventricular Tachycardia",
    "SR": "Sinus Rhythm"
}
TARGET = "Rhythm_group"
DROP_FROM_FEATURES = {"FileName", "Rhythm", "Rhythm_group"}

# ------------------------------- Data helpers --------------------------------
def read_table(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".csv"):
        return pd.read_csv(file)
    raise ValueError("Upload an Excel (.xlsx/.xls) or CSV file.")

def normalize_rhythm(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    if s in GROUPS:
        return s
    return RHYTHM_MAP.get(s, np.nan)

def prepare_dataset(raw: pd.DataFrame):
    df = raw.copy()
    if "Rhythm_group" in df.columns:
        existing = df["Rhythm_group"].astype(str).str.strip().str.upper()
        mapped_existing = existing.where(existing.isin(GROUPS))
    else:
        mapped_existing = pd.Series(np.nan, index=df.index)
    if "Rhythm" in df.columns:
        mapped_from_rhythm = df["Rhythm"].map(normalize_rhythm)
        df["Rhythm_group"] = mapped_from_rhythm.fillna(mapped_existing)
    else:
        df["Rhythm_group"] = mapped_existing
    return df

def add_engineered_features(data: pd.DataFrame):
    """Notebook-inspired ECG feature engineering; deliberately excludes target-derived features."""
    x = data.copy()
    for c in ["PatientAge", "VentricularRate", "AtrialRate", "QRSDuration",
              "QTInterval", "QTCorrected"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    pairs = [
        ("VentricularRate", "AtrialRate", "Rate_Difference", "subtract"),
        ("QTInterval", "QRSDuration", "QT_QRS_Ratio", "divide"),
        ("QTCorrected", "QRSDuration", "QTc_QRS_Ratio", "divide"),
        ("QTCorrected", "QTInterval", "QT_QTc_Difference", "subtract"),
    ]
    for a, b, name, op in pairs:
        if a in x.columns and b in x.columns:
            if op == "subtract":
                x[name] = x[a] - x[b]
            else:
                x[name] = x[a] / (x[b].replace(0, np.nan) + 1e-6)

    if "VentricularRate" in x.columns and "AtrialRate" in x.columns:
        x["Rate_Absolute_Diff"] = (x["VentricularRate"] - x["AtrialRate"]).abs()
        x["Rate_Ratio"] = x["VentricularRate"] / (x["AtrialRate"].replace(0, np.nan) + 1e-6)

    if "VentricularRate" in x.columns:
        x["Tachycardia"] = (x["VentricularRate"] > 100).astype(float)
        x["Bradycardia"] = (x["VentricularRate"] < 60).astype(float)
    if "QRSDuration" in x.columns:
        x["Wide_QRS"] = (x["QRSDuration"] > 120).astype(float)
    if "QTCorrected" in x.columns:
        x["Long_QTc"] = (x["QTCorrected"] > 450).astype(float)

    if "QTCorrected" in x.columns and "VentricularRate" in x.columns:
        x["Electrical_Stress_Index"] = x["QTCorrected"] * x["VentricularRate"]
        x["Rate_x_QTc"] = x["VentricularRate"] * x["QTCorrected"]
    if "QRSDuration" in x.columns and "QTCorrected" in x.columns:
        x["Conduction_Index"] = x["QRSDuration"] * x["QTCorrected"]
    if "PatientAge" in x.columns:
        x["Age_group"] = pd.cut(
            x["PatientAge"], bins=[0, 40, 60, 80, 120],
            labels=["young_adult", "middle_age", "senior", "elderly"],
            include_lowest=True
        ).astype("object")
        if "QTCorrected" in x.columns:
            x["Age_x_QTc"] = x["PatientAge"] * x["QTCorrected"]
        if "QRSDuration" in x.columns:
            x["Age_x_QRS"] = x["PatientAge"] * x["QRSDuration"]

    ecg_cols = [c for c in ["VentricularRate", "AtrialRate", "QRSDuration",
                            "QTInterval", "QTCorrected"] if c in x.columns]
    if len(ecg_cols) >= 2:
        x["ECG_Mean"] = x[ecg_cols].mean(axis=1)
        x["ECG_STD"] = x[ecg_cols].std(axis=1)
        x["ECG_Max"] = x[ecg_cols].max(axis=1)
        x["ECG_Min"] = x[ecg_cols].min(axis=1)
        x["ECG_Range"] = x["ECG_Max"] - x["ECG_Min"]
    for c in ["QTCorrected", "QRSDuration", "VentricularRate"]:
        if c in x.columns:
            x[f"{c}_Log"] = np.log1p(x[c].clip(lower=0))
    return x

def feature_frame(data: pd.DataFrame):
    x = add_engineered_features(data)
    # Critical leakage prevention: Rhythm and Rhythm_group (and notebook's
    # Rhythm_* one-hot columns) cannot be predictors of a target derived from Rhythm.
    remove = set(DROP_FROM_FEATURES)
    remove.update(c for c in x.columns if str(c).startswith("Rhythm_"))
    x = x.drop(columns=[c for c in remove if c in x.columns], errors="ignore")
    # Also remove identifier-like columns.
    x = x.drop(columns=[c for c in x.columns if str(c).lower() in {"id", "recordid", "patientid"}], errors="ignore")
    return x

def make_model(model_name, use_smote, random_state=42):
    if model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=350, max_depth=18, min_samples_leaf=1,
            class_weight=None if use_smote else "balanced",
            random_state=random_state, n_jobs=-1
        )
    elif model_name == "XGBoost":
        estimator = XGBClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=random_state, n_jobs=4
        )
    else:
        estimator = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
    numeric = ["number"]
    cat = ["object", "category", "bool"]
    # Use selectors at fit time, so new uploads can have a different mix of fields.
    from sklearn.compose import make_column_selector
    numeric_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipe, make_column_selector(dtype_include=cat))
    ], remainder="drop")
    steps = [("preprocess", prep)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=3)))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps)

def get_feature_importance(fitted_pipeline, top_n=25):
    clf = fitted_pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        return None
    try:
        names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
        vals = clf.feature_importances_
        return pd.DataFrame({"Feature": names, "Importance": vals}).sort_values(
            "Importance", ascending=False
        ).head(top_n)
    except Exception:
        return None

def metrics_for(y_true, y_pred, y_prob, labels):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    try:
        out["ROC-AUC (OVR weighted)"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="weighted", labels=labels
        )
    except Exception:
        out["ROC-AUC (OVR weighted)"] = np.nan
    return out

def predict_proba_aligned(pipe, X, class_ids):
    raw_prob = pipe.predict_proba(X)
    pipe_classes = pipe.named_steps["classifier"].classes_
    aligned = np.zeros((len(X), len(class_ids)))
    for j, cls in enumerate(pipe_classes):
        if cls in class_ids:
            aligned[:, list(class_ids).index(cls)] = raw_prob[:, j]
    return aligned

def excel_bytes(df):
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="ECG_Data")
    return buff.getvalue()

# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.markdown("## ❤️ ECG Rhythm Studio")
    st.caption("Interactive analytics & machine learning")
    page = st.radio("Workspace", [
        "Overview", "Data upload & cleaning", "Exploratory analysis",
        "Feature engineering", "Model lab", "Prediction", "About & methodology"
    ], label_visibility="collapsed")
    st.divider()
    upload = st.file_uploader("Upload Diagnostics dataset", type=["xlsx", "xls", "csv"])
    if upload is not None:
        try:
            if st.session_state.get("source_name") != upload.name:
                st.session_state["raw_df"] = read_table(upload)
                st.session_state["source_name"] = upload.name
                st.session_state.pop("prepared_df", None)
                st.session_state.pop("trained_bundle", None)
        except Exception as e:
            st.error(f"Could not read dataset: {e}")
    if st.session_state.get("raw_df") is not None:
        st.success(f"Loaded: {st.session_state['source_name']}")
        st.caption(f"{len(st.session_state['raw_df']):,} rows · {len(st.session_state['raw_df'].columns)} columns")
    else:
        st.info("Upload Diagnostics.xlsx or a CSV to activate the analysis.")
    st.divider()
    st.caption("Notebook rhythm mapping")
    st.markdown("**SB · AFIB · GSVT · SR**")

raw_df = st.session_state.get("raw_df")
if raw_df is not None and st.session_state.get("prepared_df") is None:
    st.session_state["prepared_df"] = prepare_dataset(raw_df)
df = st.session_state.get("prepared_df")
bundle = st.session_state.get("trained_bundle")

st.markdown("""
<div class="hero">
 <div class="eyebrow">ECG analytics · Rhythm grouping · Machine learning</div>
 <h1>ECG Rhythm Intelligence Studio</h1>
 <p>Explore diagnostic records, engineer ECG features, compare classifiers and analyse new records.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Research-use notice:</b> This app is an educational and research prototype, not a medical device. '
            'It is not validated for diagnosis, triage or treatment decisions. All outputs require qualified clinical interpretation.</div>',
            unsafe_allow_html=True)
st.write("")

# -------------------------------- Overview -----------------------------------
if page == "Overview":
    st.subheader("Your ECG analytics workspace")
    if df is not None:
        n = len(df)
        mapped = int(df[TARGET].notna().sum())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Records", f"{n:,}")
        c2.metric("Mapped rhythms", f"{mapped:,}", f"{mapped/max(n,1):.1%}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        counts = df[TARGET].value_counts().reindex(GROUPS, fill_value=0).rename_axis("Rhythm group").reset_index(name="Records")
        fig = px.bar(counts, x="Rhythm group", y="Records", color="Rhythm group",
                     title="Mapped rhythm-group distribution", text="Records")
        st.plotly_chart(fig, use_container_width=True)
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rhythm groups", "4")
        c2.metric("Model families", "3")
        c3.metric("Feature engineering", "20+")
        c4.metric("Evaluation", "Multiclass")
        st.info("Upload your Diagnostics.xlsx file in the sidebar to begin.")
    st.markdown("### Explore the workspace")
    cols = st.columns(3)
    tiles = [
        ("🧹","Data quality","Map labels, inspect missing values and duplicates, and export a cleaned workbook."),
        ("📊","Exploratory analysis","Interactive distributions, age and gender breakdowns, correlations and group comparisons."),
        ("🧪","Feature engineering","Inspect notebook-inspired heart-rate, interval, composite and statistical features."),
        ("🤖","Model lab","Train Random Forest, XGBoost and KNN; compare metrics and confusion matrices."),
        ("🔎","Prediction","Predict rhythm groups for unseen ECG records and download results."),
        ("📦","Model export","Save the complete preprocessing + classifier pipeline for later use.")
    ]
    for i,(ico,title,desc) in enumerate(tiles):
        with cols[i%3]:
            st.markdown(f"#### {ico} {title}")
            st.write(desc)

# --------------------------- Data upload & cleaning -------------------------
elif page == "Data upload & cleaning":
    st.subheader("Data upload & quality control")
    if df is None:
        st.info("Upload an Excel or CSV dataset in the sidebar.")
    else:
        a,b,c,d = st.columns(4)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        tab1,tab2,tab3 = st.tabs(["Preview & schema","Missingness","Rhythm mapping"])
        with tab1:
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            schema = pd.DataFrame({
                "Column":df.columns,
                "Type":[str(x) for x in df.dtypes],
                "Non-null":[int(df[c].notna().sum()) for c in df.columns],
                "Missing":[int(df[c].isna().sum()) for c in df.columns],
                "Unique":[int(df[c].nunique(dropna=True)) for c in df.columns]
            })
            st.dataframe(schema, use_container_width=True, hide_index=True)
        with tab2:
            miss = df.isna().sum().sort_values(ascending=False)
            miss = miss[miss>0]
            if len(miss):
                mdf = miss.rename_axis("Column").reset_index(name="Missing")
                st.plotly_chart(px.bar(mdf, x="Column", y="Missing", title="Missing values by column"), use_container_width=True)
            else:
                st.success("No missing values detected.")
            st.write("Duplicate rows:", int(df.duplicated().sum()))
            if st.checkbox("Show duplicate records"):
                st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
        with tab3:
            if "Rhythm" in df.columns:
                raw_counts = df["Rhythm"].astype(str).value_counts().rename_axis("Original rhythm").reset_index(name="Count")
                st.plotly_chart(px.bar(raw_counts, x="Original rhythm", y="Count", title="Original rhythm labels"), use_container_width=True)
            mapped_counts = df[TARGET].value_counts(dropna=False).rename_axis("Mapped group").reset_index(name="Count")
            st.dataframe(mapped_counts, use_container_width=True, hide_index=True)
            unmapped = df[df[TARGET].isna()]
            if len(unmapped):
                st.warning(f"{len(unmapped):,} records have missing or unmapped rhythm labels.")
                if "Rhythm" in unmapped:
                    st.write("Unmapped labels:", sorted(unmapped["Rhythm"].dropna().astype(str).unique().tolist()))
            else:
                st.success("All rhythm labels were mapped.")
        st.markdown("### Cleaning controls")
        col1,col2 = st.columns(2)
        with col1:
            drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
        with col2:
            drop_unmapped = st.checkbox("Exclude records without a mapped rhythm group", value=True)
        if st.button("Apply cleaning to working dataset", type="primary"):
            clean = df.copy()
            before = len(clean)
            if drop_duplicates:
                clean = clean.drop_duplicates()
            if drop_unmapped:
                clean = clean[clean[TARGET].notna()]
            st.session_state["prepared_df"] = clean.reset_index(drop=True)
            st.success(f"Working dataset updated: {before:,} → {len(clean):,} rows.")
            st.rerun()
        st.download_button("Download grouped dataset (Excel)", excel_bytes(df),
                           "Diagnostics_grouped.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------- Exploratory analysis --------------------------
elif page == "Exploratory analysis":
    st.subheader("Interactive exploratory data analysis")
    if df is None:
        st.info("Upload a dataset first.")
    else:
        valid = df[df[TARGET].notna()].copy()
        num_cols = valid.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in valid.select_dtypes(exclude=np.number).columns if c not in [TARGET,"FileName","Rhythm"]]
        t1,t2,t3,t4 = st.tabs(["Rhythm groups","Numeric features","Demographics","Relationships"])
        with t1:
            counts = valid[TARGET].value_counts().reindex(GROUPS, fill_value=0).rename_axis("Group").reset_index(name="Records")
            fig = px.bar(counts, x="Group", y="Records", color="Group", text="Records", title="Rhythm-group counts")
            st.plotly_chart(fig, use_container_width=True)
            share = counts.copy()
            share["Percentage"] = share["Records"]/max(share["Records"].sum(),1)*100
            st.dataframe(share, hide_index=True, use_container_width=True)
            if "Rhythm" in valid.columns:
                cross = pd.crosstab(valid["Rhythm"], valid[TARGET])
                st.markdown("**Original label → grouped label cross-tabulation**")
                st.dataframe(cross, use_container_width=True)
        with t2:
            if num_cols:
                col = st.selectbox("Choose a numerical variable", num_cols, index=num_cols.index("VentricularRate") if "VentricularRate" in num_cols else 0)
                chart = st.selectbox("Chart type", ["Histogram","Box plot","Violin plot"])
                if chart == "Histogram":
                    fig = px.histogram(valid, x=col, color=TARGET, marginal="box", barmode="overlay", opacity=.65)
                elif chart == "Box plot":
                    fig = px.box(valid, y=col, color=TARGET, points="outliers")
                else:
                    fig = px.violin(valid, y=col, x=TARGET, color=TARGET, box=True, points=False)
                fig.update_layout(title=f"{chart}: {col}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(valid.groupby(TARGET)[col].describe().round(2), use_container_width=True)
            else:
                st.info("No numerical columns found.")
        with t3:
            if "PatientAge" in valid.columns:
                st.plotly_chart(px.histogram(valid, x="PatientAge", color=TARGET, marginal="box",
                                             title="Patient age distribution by rhythm group"), use_container_width=True)
                st.plotly_chart(px.box(valid, x=TARGET, y="PatientAge", color=TARGET,
                                       title="Age by rhythm group"), use_container_width=True)
            if "Gender" in valid.columns:
                gender = pd.crosstab(valid["Gender"], valid[TARGET]).reset_index()
                st.plotly_chart(px.bar(gender, x="Gender", y=GROUPS, barmode="stack",
                                       title="Gender vs rhythm group"), use_container_width=True)
                st.dataframe(pd.crosstab(valid["Gender"], valid[TARGET]), use_container_width=True)
        with t4:
            candidates = [c for c in ["VentricularRate","AtrialRate","QRSDuration","QTInterval","QTCorrected","RAxis","TAxis"] if c in num_cols]
            if len(candidates)>=2:
                selected = st.multiselect("Select features for correlation", candidates, default=candidates[:min(6,len(candidates))])
                if len(selected)>=2:
                    corr = valid[selected].corr(numeric_only=True)
                    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                    zmin=-1, zmax=1, title="Feature correlation matrix")
                    st.plotly_chart(fig, use_container_width=True)
            if len(num_cols)>=2:
                xcol = st.selectbox("X axis", num_cols, index=0)
                ycol = st.selectbox("Y axis", num_cols, index=min(1,len(num_cols)-1))
                st.plotly_chart(px.scatter(valid.sample(min(2500,len(valid)), random_state=42),
                                           x=xcol,y=ycol,color=TARGET,opacity=.6,
                                           title=f"{ycol} vs {xcol}"), use_container_width=True)

# ----------------------------- Feature engineering -------------------------
elif page == "Feature engineering":
    st.subheader("ECG feature engineering")
    if df is None:
        st.info("Upload a dataset first.")
    else:
        engineered = add_engineered_features(df)
        engineered_features = [
            "Rate_Difference","Rate_Ratio","Rate_Absolute_Diff","QT_QRS_Ratio",
            "QTc_QRS_Ratio","QT_QTc_Difference","Tachycardia","Bradycardia",
            "Wide_QRS","Long_QTc","Electrical_Stress_Index","Conduction_Index",
            "Age_x_QTc","Age_x_QRS","Rate_x_QTc","ECG_Mean","ECG_STD",
            "ECG_Max","ECG_Min","ECG_Range","QTCorrected_Log","QRSDuration_Log",
            "VentricularRate_Log","Age_group"
        ]
        available = [c for c in engineered_features if c in engineered.columns]
        st.write("The feature set follows the notebook's engineering ideas: rate differences and ratios, ECG interval ratios, binary indicators, composite indices, age interactions, summary statistics and log transforms.")
        st.metric("Engineered features available", len(available))
        selected = st.multiselect("Choose engineered features to inspect", available, default=available)
        if selected:
            st.dataframe(engineered[selected].head(100), use_container_width=True)
            numeric_selected = engineered[selected].select_dtypes(include=np.number).columns.tolist()
            if numeric_selected:
                feature = st.selectbox("Plot feature", numeric_selected)
                st.plotly_chart(px.histogram(engineered, x=feature, color=TARGET if TARGET in engineered else None,
                                             title=f"Distribution of {feature}"), use_container_width=True)
        st.download_button("Download engineered dataset", excel_bytes(engineered),
                           "Diagnostics_engineered.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("### Leakage-safe predictor set")
        st.info("For a target derived from Rhythm, the app excludes Rhythm, Rhythm_group, FileName, and every Rhythm_* one-hot column from predictors. Beat and Gender remain eligible predictors.")
        X_preview = feature_frame(df)
        st.write(f"Predictor columns: **{X_preview.shape[1]}**")
        with st.expander("Show predictor columns"):
            st.write(X_preview.columns.tolist())

# -------------------------------- Model lab ----------------------------------
elif page == "Model lab":
    st.subheader("Model development & comparison")
    if df is None:
        st.info("Upload a dataset first.")
    else:
        train_df = df[df[TARGET].notna()].copy()
        if train_df[TARGET].nunique() < 2:
            st.error("At least two mapped rhythm groups are required for classification.")
        else:
            left,right = st.columns(2)
            with left:
                models_selected = st.multiselect("Models to train",
                    ["Random Forest","XGBoost","KNN"], default=["Random Forest","XGBoost","KNN"])
                test_size = st.slider("Test set proportion", .10, .35, .20, .05)
                use_smote = st.checkbox("Apply SMOTE to training split only", value=False,
                                        help="SMOTE is applied inside the training pipeline, never to the held-out test set.")
            with right:
                random_state = st.number_input("Random seed", 0, 99999, 42)
                xgb_estimators = st.slider("Tree count (RF / XGBoost)", 100, 600, 350, 50)
                show_importance = st.checkbox("Show feature importance when supported", True)
            st.caption(f"Training records: {len(train_df):,} · target classes: {train_df[TARGET].nunique()}")
            if use_smote and train_df[TARGET].value_counts().min() <= 3:
                st.warning("SMOTE needs at least 4 records in every class with the current k-neighbours setting.")
            if st.button("Train and evaluate selected models", type="primary", disabled=not models_selected):
                try:
                    X = feature_frame(train_df)
                    y = train_df[TARGET].astype(str)
                    labels = sorted(y.unique().tolist())
                    label_encoder = LabelEncoder().fit(labels)
                    y_encoded = pd.Series(label_encoder.transform(y), index=y.index)
                    class_ids = list(range(len(labels)))
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=int(random_state), stratify=y_encoded
                    )
                    results, trained = [], {}
                    progress = st.progress(0)
                    status = st.empty()
                    for i, name in enumerate(models_selected):
                        status.write(f"Training {name}…")
                        pipe = make_model(name, use_smote, int(random_state))
                        if name in ["Random Forest","XGBoost"]:
                            pipe.named_steps["classifier"].set_params(n_estimators=int(xgb_estimators))
                        if use_smote:
                            min_count = int(y_train.value_counts().min())
                            pipe.set_params(smote__k_neighbors=min(3, min_count-1))
                        pipe.fit(X_train, y_train)
                        pred = pipe.predict(X_test)
                        prob = predict_proba_aligned(pipe, X_test, class_ids)
                        met = metrics_for(y_test, pred, prob, class_ids)
                        met["Model"] = name
                        results.append(met)
                        trained[name] = {
                            "pipeline": pipe, "labels": labels, "class_ids": class_ids,
                            "feature_columns": list(X.columns),
                            "metrics": met,
                            "y_test": y_test.to_numpy(), "y_pred": pred,
                            "y_prob": prob, "X_test": X_test
                        }
                        progress.progress((i+1)/len(models_selected))
                    st.session_state["model_results"] = pd.DataFrame(results).set_index("Model")
                    st.session_state["trained_models"] = trained
                    st.session_state["active_model"] = max(results, key=lambda x: x["F1 (weighted)"])["Model"]
                    status.success("Training and evaluation completed.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.exception(e)

            results = st.session_state.get("model_results")
            trained_models = st.session_state.get("trained_models")
            if results is not None and trained_models:
                st.markdown("### Performance comparison")
                st.dataframe(results.style.format("{:.4f}"), use_container_width=True)
                metric_cols = [c for c in ["Accuracy","Precision (weighted)","Recall (weighted)","F1 (weighted)","F1 (macro)","ROC-AUC (OVR weighted)"] if c in results]
                fig = px.bar(results.reset_index().melt(id_vars="Model", value_vars=metric_cols,
                            var_name="Metric",value_name="Score"),
                            x="Model",y="Score",color="Metric",barmode="group",
                            title="Classifier performance comparison", range_y=[0,1.05])
                st.plotly_chart(fig, use_container_width=True)
                active = st.selectbox("Inspect model", list(trained_models.keys()),
                                      index=list(trained_models.keys()).index(st.session_state.get("active_model")))
                st.session_state["active_model"] = active
                detail = trained_models[active]
                cm = confusion_matrix(detail["y_test"], detail["y_pred"], labels=detail["class_ids"])
                fig = px.imshow(cm, text_auto=True, x=detail["labels"], y=detail["labels"],
                                labels=dict(x="Predicted",y="Actual",color="Records"),
                                color_continuous_scale="Blues", title=f"{active}: confusion matrix")
                st.plotly_chart(fig, use_container_width=True)
                report = classification_report(detail["y_test"], detail["y_pred"],
                    labels=detail["class_ids"], target_names=detail["labels"],
                    output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).T, use_container_width=True)
                if show_importance:
                    imp = get_feature_importance(detail["pipeline"])
                    if imp is not None:
                        st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance",y="Feature",
                                              orientation="h",title=f"{active}: top feature importance"),
                                        use_container_width=True)
                    else:
                        st.caption("Feature importance is not directly available for KNN.")
                # CV is intentionally optional to avoid a long deployment run.
                if st.checkbox("Run 5-fold cross-validation for selected model"):
                    from sklearn.model_selection import cross_val_score, StratifiedKFold
                    with st.spinner("Running 5-fold cross-validation…"):
                        cv_pipe = make_model(active, use_smote, int(random_state))
                        scores = cross_val_score(cv_pipe, X, y_encoded, cv=StratifiedKFold(5,shuffle=True,random_state=int(random_state)),
                                                 scoring="f1_weighted", n_jobs=1)
                    st.metric("Mean CV weighted F1", f"{scores.mean():.4f}", f"± {scores.std():.4f}")
                    st.write(pd.DataFrame({"Fold":range(1,6),"Weighted F1":scores}))
                if st.button("Save selected model as active prediction model"):
                    st.session_state["trained_bundle"] = {
                        "pipeline": detail["pipeline"],
                        "labels": detail["labels"], "class_ids": detail["class_ids"],
                        "feature_columns": detail["feature_columns"],
                        "target": TARGET,
                        "rhythm_map": RHYTHM_MAP,
                        "model_name": active
                    }
                    st.success(f"{active} saved for the Prediction workspace.")
                bundle_bytes = io.BytesIO()
                joblib.dump({
                    "pipeline": detail["pipeline"], "labels": detail["labels"],
                    "class_ids": detail["class_ids"], "feature_columns": detail["feature_columns"], "target": TARGET,
                    "rhythm_map": RHYTHM_MAP, "model_name": active
                }, bundle_bytes)
                st.download_button("Download selected model bundle (.joblib)", bundle_bytes.getvalue(),
                                   f"ecg_{active.lower().replace(' ','_')}_bundle.joblib",
                                   "application/octet-stream")

# -------------------------------- Prediction --------------------------------
elif page == "Prediction":
    st.subheader("Predict rhythm group for new records")
    if bundle is None:
        st.info("Train a model in Model lab and save it as the active prediction model. You can also upload a previously exported .joblib bundle below.")
        model_upload = st.file_uploader("Load saved model bundle", type=["joblib","pkl"], key="model_bundle")
        if model_upload is not None and st.button("Load model bundle"):
            try:
                loaded = joblib.load(model_upload)
                required = {"pipeline","labels","feature_columns"}
                if not required.issubset(loaded):
                    raise ValueError("The uploaded file is not an ECG Rhythm Studio model bundle.")
                st.session_state["trained_bundle"] = loaded
                st.success("Model bundle loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load model: {e}")
    else:
        st.success(f"Active model: {bundle.get('model_name','Loaded model')}")
        st.write("Upload a CSV/Excel file of unseen ECG records. The app will ignore any Rhythm/Rhythm_group columns in the uploaded data.")
        new_file = st.file_uploader("New ECG records", type=["xlsx","xls","csv"], key="new_records")
        if new_file is not None:
            try:
                new_data = read_table(new_file)
                Xnew = feature_frame(new_data)
                # Reindex to the model's training columns; absent fields become missing and are imputed.
                Xnew = Xnew.reindex(columns=bundle["feature_columns"])
                if st.button("Run predictions", type="primary"):
                    pipe = bundle["pipeline"]
                    pred_ids = pipe.predict(Xnew).astype(int)
                    class_ids = bundle.get("class_ids", list(range(len(bundle["labels"]))))
                    prob = predict_proba_aligned(pipe, Xnew, class_ids)
                    pred = [bundle["labels"][int(i)] for i in pred_ids]
                    out = new_data.copy()
                    out["Predicted_Rhythm_Group"] = pred
                    out["Predicted_Rhythm_Name"] = [GROUP_NAMES.get(str(x),str(x)) for x in pred]
                    for i,label in enumerate(bundle["labels"]):
                        out[f"Probability_{label}"] = prob[:,i]
                    st.session_state["prediction_output"] = out
                    st.session_state["prediction_probs"] = prob
                    st.session_state["prediction_labels"] = bundle["labels"]
                output = st.session_state.get("prediction_output")
                if output is not None:
                    st.markdown("### Prediction results")
                    st.dataframe(output, use_container_width=True, hide_index=True)
                    st.download_button("Download predictions (Excel)", excel_bytes(output),
                                       "ECG_rhythm_predictions.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.download_button("Download predictions (CSV)", output.to_csv(index=False).encode(),
                                       "ECG_rhythm_predictions.csv","text/csv")
                    probs = st.session_state.get("prediction_probs")
                    labs = st.session_state.get("prediction_labels")
                    if probs is not None and len(probs):
                        plot_df = pd.DataFrame(probs,columns=labs)
                        plot_df["Record"] = [str(i+1) for i in range(len(plot_df))]
                        st.plotly_chart(px.bar(plot_df.melt(id_vars="Record",var_name="Rhythm group",value_name="Score"),
                                               x="Record",y="Score",color="Rhythm group",barmode="stack",
                                               title="Predicted class-score distribution"), use_container_width=True)
            except Exception as e:
                st.error(f"Could not process prediction file: {e}")
        with st.expander("Predict a single record manually"):
            st.caption("Enter the available base measurements. Missing fields are handled by the trained pipeline.")
            manual = {}
            numeric_fields = ["PatientAge","VentricularRate","AtrialRate","QRSDuration","QTInterval","QTCorrected",
                              "RAxis","TAxis","QRSCount","QOnset","QOffset","TOffset"]
            cols = st.columns(3)
            for i,col in enumerate(numeric_fields):
                with cols[i%3]:
                    manual[col] = st.number_input(col, value=0.0, key=f"manual_{col}")
            manual["Gender"] = st.selectbox("Gender",["FEMALE","MALE"],key="manual_gender")
            manual["Beat"] = st.text_input("Beat description (optional)",value="NONE",key="manual_beat")
            if st.button("Predict this record"):
                row = pd.DataFrame([manual])
                xx = feature_frame(row).reindex(columns=bundle["feature_columns"])
                class_ids = bundle.get("class_ids", list(range(len(bundle["labels"]))))
                pp = predict_proba_aligned(bundle["pipeline"],xx,class_ids)[0]
                ix = int(np.argmax(pp))
                st.metric("Predicted rhythm group", f"{bundle['labels'][ix]} · {GROUP_NAMES.get(bundle['labels'][ix],'')}")
                st.plotly_chart(px.bar(pd.DataFrame({"Rhythm group":bundle["labels"],"Score":pp}),
                                       x="Rhythm group",y="Score",title="Model class scores"), use_container_width=True)

# -------------------------- About / methodology -----------------------------
elif page == "About & methodology":
    st.subheader("About this application")
    st.write("This Streamlit application is based on the uploaded ECG diagnostics notebook and its data-analysis workflow.")
    with st.expander("Rhythm grouping used in the notebook", expanded=True):
        st.table(pd.DataFrame([
            ["SB","SB","Sinus Bradycardia"],
            ["AFIB / AF","AFIB","Atrial Fibrillation/Flutter"],
            ["ST / SVT / AT / AVNRT / AVRT / SAAWR","GSVT","Generalized Supraventricular Tachycardia"],
            ["SR / SI / SA","SR","Sinus Rhythm"]
        ],columns=["Original rhythm label(s)","Grouped label","Description"]))
    with st.expander("Dataset fields seen in the notebook"):
        st.write(["FileName","Rhythm","Beat","PatientAge","Gender","VentricularRate","AtrialRate",
                  "QRSDuration","QTInterval","QTCorrected","RAxis","TAxis","QRSCount",
                  "QOnset","QOffset","TOffset","Rhythm_group"])
    with st.expander("Feature engineering"):
        st.write("The app adds rate differences/ratios, QT/QRS ratios, tachycardia/bradycardia flags, wide-QRS and long-QTc flags, composite indices, age interactions, ECG summary statistics and log transforms.")
    with st.expander("Model methodology and safeguards"):
        st.markdown("""
- The target is `Rhythm_group`, derived from the notebook's original `Rhythm` mapping.
- `Rhythm`, `Rhythm_group`, `FileName` and `Rhythm_*` columns are excluded from predictors to prevent target leakage.
- Preprocessing (median/mode imputation, scaling and one-hot encoding) is fitted inside the training pipeline.
- The test split is held out from fitting. Optional SMOTE is applied only within the training pipeline.
- Models: Random Forest, XGBoost and K-Nearest Neighbours.
- Evaluation: accuracy, weighted precision/recall/F1, macro F1, MCC, multiclass ROC-AUC where available, confusion matrix and classification report.
        """)
    with st.expander("Deployment notes"):
        st.markdown("""
Use Python 3.11 for Streamlit Community Cloud. TensorFlow is not required for this project.
Upload the dataset at runtime; do not commit patient-level data or model artifacts to a public GitHub repository.
        """)
    st.markdown("### Responsible use")
    st.write("This is not a clinical decision-support system. Model performance on a notebook split does not establish generalisability, clinical validity or patient safety. Validate externally and obtain appropriate institutional approvals before any real-world use.")

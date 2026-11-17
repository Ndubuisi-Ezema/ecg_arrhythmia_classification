# ECG Rhythm Intelligence Studio

Interactive Streamlit application based on `Project1_(3)_(1).ipynb`.

## Features
- Upload Excel (`.xlsx`, `.xls`) or CSV ECG diagnostics data
- Map original rhythm labels into SB, AFIB, GSVT and SR
- Data-quality review: schema, missingness, duplicates and unmapped labels
- Interactive EDA: rhythm distributions, age/gender analyses, histograms, box/violin plots, correlations and scatter plots
- Notebook-inspired ECG feature engineering
- Leakage-safe training and evaluation of Random Forest, XGBoost and KNN
- Optional SMOTE inside the training pipeline only
- Metrics: accuracy, weighted precision/recall/F1, macro F1, MCC and multiclass ROC-AUC when available
- Confusion matrix, classification report and feature importance
- Download a reusable preprocessing + model bundle
- Batch and single-record prediction with CSV/Excel export

## Dataset
Expected source fields include:
`FileName`, `Rhythm`, `Beat`, `PatientAge`, `Gender`, `VentricularRate`, `AtrialRate`,
`QRSDuration`, `QTInterval`, `QTCorrected`, `RAxis`, `TAxis`, `QRSCount`, `QOnset`,
`QOffset`, `TOffset`. The app derives `Rhythm_group` from `Rhythm`.

Rhythm map:
- SB -> SB
- AFIB, AF -> AFIB
- ST, SVT, AT, AVNRT, AVRT, SAAWR -> GSVT
- SR, SI, SA -> SR

## Run locally
Python 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to the root of a GitHub repository.
2. Create an app in Streamlit Community Cloud using branch `main`, entrypoint `app.py`.
3. Select Python 3.11 in Advanced settings.
4. Deploy. This project does not need TensorFlow.

## Important modelling safeguard
The notebook one-hot encodes the original `Rhythm` column while using `Rhythm_group` (derived from `Rhythm`) as the target. That would leak target information. This app explicitly excludes `Rhythm`, `Rhythm_group`, `FileName`, and `Rhythm_*` dummy columns from the predictors. Preprocessing is fitted on training data only; optional SMOTE runs only inside the training pipeline.

## Safety and privacy
Research/education only; not validated for diagnosis or treatment decisions. Do not upload identifiable patient data to a hosted service unless your institution has approved the data processing, hosting and retention arrangements.

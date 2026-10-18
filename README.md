# Heart Disease Risk Prediction System

A Streamlit decision-support application that estimates the likelihood of heart disease from clinical measurements. It provides a single-patient prediction form, batch CSV scoring, model-quality metrics, and diagnostic visualizations for the included trained model.

> **Clinical disclaimer:** This project is for educational and research use only. It is not a diagnostic device and must not replace a qualified clinician's judgement, full patient history, or confirmatory testing.

## Highlights

- Real-time prediction with an estimated risk probability and configurable decision threshold.
- Batch CSV inference with downloadable results.
- A trained `RandomForestClassifier` pipeline with median imputation and one-hot encoding.
- Model-quality dashboard with accuracy, ROC-AUC, PR-AUC, precision, recall, F1, specificity, calibration, and feature importance.
- Diagnostic gallery containing ROC, precision-recall, calibration, learning-curve, threshold, feature-importance, probability-distribution, and confusion-matrix plots.

## Included model

The application loads `FINAL_MODEL.joblib`, which was trained on the UCI Cleveland Heart Disease data. It expects the following 13 features:

| Feature | Meaning |
| --- | --- |
| `age` | Age in years |
| `sex` | Sex code: 0=female, 1=male |
| `cp` | Chest pain type: 1-4 |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar >120 mg/dL: 0=no, 1=yes |
| `restecg` | Resting ECG result: 0-2 |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina: 0=no, 1=yes |
| `oldpeak` | Exercise-induced ST depression |
| `slope` | Peak exercise ST-segment slope: 1-3 |
| `ca` | Number of major vessels: 0-4 |
| `thal` | Thalassemia code: 3, 6, or 7 |

The saved evaluation results report **89.13% accuracy**, **94.86% ROC-AUC**, and **87.80% F1** on the held-out test set using a **0.585** decision threshold.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Ayaanwanii/Heart-Disease-Risk-Prediction-System.git
cd Heart-Disease-Risk-Prediction-System
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with `.venv\\Scripts\\activate`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Batch prediction CSV

Upload a CSV containing every feature listed above. For example:

```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
54,1,3,130,246,0,1,150,0,1.0,2,0,3
```

The downloaded output retains the input columns and adds `risk_probability`, `risk_percent`, `prediction`, and `threshold_used`.

## Reproducing the UCI model

`model.py` trains the UCI-based Random Forest pipeline and regenerates the application artifacts. It uses the local `heart_disease_dataset_cache.csv` when present; otherwise it tries the UCI repository and then the direct Cleveland dataset URL.

```bash
python model.py
```

`build_final_model.py` is a separate experimental script for the included cardiovascular dataset (`cardio_train.csv`). Its output is written to an `outputs/` directory and is not the model currently served by `app.py`.

## Project structure

```text
app.py                                  # Streamlit application
model.py                                # UCI model training and diagnostics
FINAL_MODEL.joblib                      # Model loaded by the application
FINAL_MODEL_results.csv                 # Saved evaluation metrics
FINAL_MODEL_*.png                       # Diagnostic charts
heart_disease_dataset_cache.csv         # Cached UCI Cleveland data
build_final_model.py                    # Separate cardiovascular-data experiment
cardio_train.csv                        # Dataset for that experiment
requirements.txt                        # Python dependencies
```

## Deployment

This repository is ready for [Streamlit Community Cloud](https://share.streamlit.io/): select the repository, choose branch `main`, and set the main file path to `app.py`. The included `requirements.txt` is used for dependency installation.

## Data source

The served model is based on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), specifically the Cleveland data representation.

## License

No license has been specified. Add a license before redistributing or reusing the project beyond its intended scope.

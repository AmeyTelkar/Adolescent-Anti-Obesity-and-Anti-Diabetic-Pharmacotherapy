# Temporal Adolescent Anti-Obesity and Anti-Diabetic Pharmacotherapy

## Safety of Anti-Obesity and Anti-Diabetic Medications in Adolescents: A Disproportionality Analysis and Machine-Learning Validation from 2021-2025 on the Basis of the FAERS Database

[![Journal: PLOS ONE](https://img.shields.io/badge/Journal-PLOS%20ONE-blue)](https://journals.plos.org/plosone/)
[![Data: FDA FAERS](https://img.shields.io/badge/Data-FDA%20FAERS-green)](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22767355-blue)](https://doi.org/10.5281/zenodo.22767355)

**Version: V17.1 (Final PLOS ONE Submission)**

---

## Abstract

### Background
Adolescent obesity and type 2 diabetes have risen sharply, driving increased prescribing of anti-obesity and anti-diabetic agents in patients aged 12-17 years, many prescribed off-label with limited post-marketing safety data. This study aimed to analyse adverse events (AEs) associated with these medications in adolescents using the FDA Adverse Event Reporting System (FAERS).

### Methods
AE reports for adolescents aged 12-17 years were extracted from FAERS (2021Q1-2025Q4). Fourteen obesity-related and 10 diabetes-related medications were included. AEs were classified using MedDRA (version 28.1) at the Preferred Term level. Signal detection used four disproportionality methods (ROR, PRR, IC, EBGM), complemented by an XGBoost classifier for case-level outcome triage.

### Results
From 7,612,804 raw FAERS records, 403,278 adolescent reports were identified (11,701 obesity-related panel; 5,208 diabetes-related panel; ~54% female; mean age 14.57 years). Key disproportionality signals included metformin-lactic acidosis (ROR = 61.22, 95% CI 25.21-148.68, N = 214), atorvastatin-myalgia (ROR = 16.89, N = 46), semaglutide-optic ischaemic neuropathy (ROR = 439.23, N = 14, EBGM = 18.79), and dapagliflozin-cardiac failure (ROR = 40.24, N = 30). The XGBoost model achieved a primary chronological temporal Macro-F1 of **0.402** and a conventional random-split Macro-F1 of **0.635** on the obesity-related panel, significantly outperforming all baselines (McNemar's p < 0.001).

### Conclusion
This study provides a comprehensive assessment combining broad adolescent medication panels, multi-metric disproportionality analysis, and case-level machine learning outcome triage. The findings confirm known signals while identifying reporting patterns warranting further investigation, and may inform clinical practice and regulatory monitoring for this vulnerable population.

---

## Repository Structure

### Root Directory

| File / Folder | Description |
|---|---|
| `README.md` | This file |
| `regen_figs.py` | Script to regenerate all manuscript figures |
| `dataset/` | Training data and ML scripts (4 sub-folders) |
| `Submission_Figures/` | All manuscript figures (EPS + PNG) |
| `Supplementary_Information/` | Supporting tables and checklists (S1-S9) |

### `dataset/` - Training Data and ML Scripts

| Sub-folder | Description |
|---|---|
| `14 Columns Model/` | **Primary model** (leakage-free, 14 features). Contains datasets for all 4 cohorts + training scripts |
| `15 Columns Leakage/` | Leakage demonstration (includes `severity_score`). Documents the accuracy drop |
| `15 Columns Model with Source Quarter/` | **Temporal evaluation** model (chronological train/val/test split). Contains `train_obesity_all14_temporal.py` (Macro-F1 0.402 / 0.635) |
| `16 Columns Leakage with Source Quarter/` | Leakage demonstration with temporal column |

Each sub-folder contains:
- **4 cohort datasets** (`.xlsx`): ObesityAll 14 drugs, Obesity Selected 4, DiabeticsAll 10, Diabetics Selected 4
- **4 training scripts** (`.py`): One per cohort
- `ML_Multiclass_Model_Comparison_Results.xlsx` - Model comparison results
- `MLDimensionalityComparison46_32_14_Cols.xlsx` - Feature dimensionality analysis

### `Submission_Figures/` - Manuscript Figures

| File | Description |
|---|---|
| `Fig1.eps` / `Fig1.png` | STROBE-compliant study flowchart |
| `Fig2.eps` / `Fig2.png` | SOC-level AE distribution bar chart (MedDRA 28.1) |
| `Fig3.eps` / `Fig3.png` | Forest plots: metformin, atorvastatin, semaglutide |
| `Fig4.eps` / `Fig4.png` | Forest plots: dapagliflozin, glargine, tirzepatide, empagliflozin |
| `S1_Fig_Detailed_Flowchart.eps` / `.png` | Supplementary detailed flowchart |
| `S2_Fig_SHAP_Importance.eps` / `.png` | Supplementary SHAP feature importance |

### `Supplementary_Information/` - Supporting Tables

| File | Description |
|---|---|
| `S1_Table_STROBE_Checklist.docx` | STROBE checklist for observational studies |
| `S2_Table_TRIPOD_AI_Checklist.docx` | TRIPOD+AI checklist for prediction model development |
| `S3_Table_READUS_PV_Checklist.docx` | READUS-PV checklist for disproportionality analysis |
| `S4_Table_Pipeline_Specification.docx` | Complete preprocessing pipeline specification |
| `S5_Table_Hyperparameter_Search_Space.docx` | Hyperparameter search space and optimal configurations |
| `S6_Table_Predictor_Leakage_Audit.docx` | Predictor specification and data-leakage audit (17 features) |
| `S7_Table_Temporal_Shift.docx` | Temporal dataset shift analysis (train/validation/test metrics) |
| `S8_Table_Complete_Signal_Detection_FULL.xlsx` | **Complete signal detection results** - 2,098 drug-event pairs, 360 signals, with ROR/PRR/IC/EBGM + four-metric concordance flags |
| `S9_Table_Quarterly_Reporting_Volume.docx` | Quarter-wise FAERS reporting volume (2021Q1-2025Q4) |

---

## Study Design

### Data Source
- **Database**: FDA Adverse Event Reporting System (FAERS)
- **Period**: 2021Q1 - 2025Q4 (20 consecutive quarters)
- **Population**: Adolescents aged 12-17 years
- **Total raw records**: 7,612,804
- **After deduplication**: 6,985,217 unique cases
- **Age-eligible**: 403,278 adolescent reports

### Drug Panels

| Panel | Drugs | N (reports) |
|---|---|---|
| **Broad Obesity (14 drugs)** | Metformin, atorvastatin, lisinopril, losartan, dapagliflozin, insulin (regular, aspart, glargine, lispro), semaglutide, empagliflozin, tirzepatide | 11,701 |
| **Selected Obesity (4 drugs)** | Metformin, atorvastatin, lisinopril, losartan | 10,701 |
| **Broad Diabetes (10 drugs)** | Metformin, dapagliflozin, insulin (aspart, regular, lispro, glargine), semaglutide, empagliflozin, tirzepatide | 5,208 |
| **Selected Diabetes (4 drugs)** | Semaglutide, empagliflozin, tirzepatide, dapagliflozin | 342 |

### Signal Detection Methods

| Method | Criteria |
|---|---|
| **ROR** (Reporting Odds Ratio) | 95% CI lower bound > 1, N >= 3 |
| **PRR** (Proportional Reporting Ratio) | PRR >= 2, chi-squared >= 4, N >= 3 |
| **IC** (Information Component, BCPNN) | IC025 > 0 |
| **EBGM** (Empirical Bayes Geometric Mean) | EB05 >= 2 |

### Machine Learning Component
- **Model**: XGBoost (leakage-aware, 14-column model)
- **Features**: 13 case-level features (age, sex, weight, drug sequence, route, role code, drug name, RxCUI, dose amount/unit/form, indication, AE PT)
- **Target**: Outcome severity (DE/LT/HO/DS/RI/OT)
- **Split**: Chronological (Train: 2021Q1-2023Q4, Validation: 2024, Test: 2025)
- **Performance**:
  - **Temporal (chronological) Macro-F1**: **0.402**
  - **Random-split Macro-F1**: **0.635**
  - **Validation (2024) Macro-F1**: 0.338
- **Class-specific recall (temporal test)**:
  - Death: 0.40 | Life-Threatening: 0.07 | Hospitalization: 0.46 | Disability: 0.06 | Other: 0.86

---

## Key Findings

### Strongest Disproportionality Signals (ROR, Verified from Raw Data)

| Drug | Adverse Event (PT) | N | ROR | 95% CI |
|---|---|---|---|---|
| Empagliflozin | Interstitial lung disease | 4 | 475.55 | 52.82-4281.55 |
| Semaglutide | Optic ischaemic neuropathy | 14 | 439.23 | 99.33-1942.26 |
| Empagliflozin | Hepatic function abnormal | 6 | 361.77 | 72.38-1808.09 |
| Empagliflozin | Rash papular | 6 | 241.17 | 59.71-974.02 |
| Dapagliflozin | Nephrotic syndrome | 6 | 162.46 | 19.53-1351.58 |
| Dapagliflozin | Nephritis | 4 | 107.95 | 12.05-967.28 |
| Tirzepatide | Intestinal obstruction | 4 | 106.50 | 28.27-401.26 |
| Insulin Glargine | Oedema | 6 | 90.01 | 25.26-320.75 |
| Dapagliflozin | Body mass index increased | 7 | 63.28 | 16.32-245.30 |
| Metformin | Lactic acidosis | 214 | 61.22 | 25.21-148.68 |
| Dapagliflozin | Cardiac failure | 30 | 40.24 | 22.90-70.71 |
| Tirzepatide | Suicidal ideation | 4 | 38.02 | 12.34-117.09 |
| Tirzepatide | Pancreatitis | 6 | 33.78 | 13.57-84.08 |
| Atorvastatin | Myalgia | 46 | 16.89 | 8.94-31.92 |

### McNemar's Test Results (XGBoost vs. Baselines)

| Comparison | Discordant (a) | Discordant (b) | Chi-squared | p-value |
|---|---|---|---|---|
| XGBoost vs. Majority Class | 525 | 138 | 224.73 | < 0.001 |
| XGBoost vs. Logistic Regression | 511 | 130 | 225.27 | < 0.001 |
| XGBoost vs. Decision Tree | 486 | 132 | 201.63 | < 0.001 |
| XGBoost vs. Random Forest | 289 | 161 | 35.84 | < 0.001 |
| XGBoost vs. Seriousness Rules | 242 | 473 | 73.99 | < 0.001 |

---

## How to Reproduce

1. **Data**: Download quarterly FAERS ASCII files from [FDA FAERS](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard) for 2021Q1-2025Q4.
2. **Preprocessing**: Follow the pipeline specification in S4 Table (deduplication, age filtering, drug normalisation, MedDRA coding).
   > **Note**: MedDRA version 28.1 was used for SOC-level classification in this study. Due to proprietary licensing by the MSSO, the MedDRA Desktop Browser cannot be distributed in this repository. Researchers must obtain a valid license from [MedDRA (MSSO)](https://www.meddra.org/) to reproduce the exact coding.
3. **Signal detection**: Apply four disproportionality methods (ROR, PRR, IC, EBGM) as described in the manuscript.
4. **ML training**: Use scripts in `dataset/14 Columns Model/` for primary models, or `dataset/15 Columns Model with Source Quarter/` for temporal-split evaluation.
5. **Figure regeneration**: Run `regen_figs.py` to regenerate all manuscript figures from the signal detection outputs.

---

## Reporting Checklists

This study adheres to three reporting guidelines:
- **STROBE** - Strengthening the Reporting of Observational Studies in Epidemiology
- **TRIPOD+AI** - Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (AI extension)
- **READUS-PV** - Reporting of disproportionality analyses in pharmacovigilance

---

## Citation

> Telkar A, Telkar A, Javalgikar A, Madanwale N, Ruikar D, Baligar P. Safety of anti-obesity and anti-diabetic medications in adolescents: A disproportionality analysis and machine-learning validation from 2021-2025 on the basis of the FAERS database. *PLOS ONE*. 2026 (submitted).

---

## Authors

| Author | Affiliation | ORCID |
|---|---|---|
| **Atherv Telkar** | School of Computing, MIT Vishwaprayag University, Solapur | [0009-0002-2580-6498](https://orcid.org/0009-0002-2580-6498) |
| **Amey Telkar** | School of Computing, MIT Vishwaprayag University, Solapur | [0009-0000-5973-4553](https://orcid.org/0009-0000-5973-4553) |
| Akshay Javalgikar | School of Pharmacy, MIT Vishwaprayag University, Solapur | - |
| Nitin Madanwale | School of Pharmacy, MIT Vishwaprayag University, Solapur | - |
| Darshan Ruikar | School of Computing, MIT Vishwaprayag University, Solapur | - |
| Preethi Baligar | School of Computing, MIT Vishwaprayag University, Solapur | - |

**Corresponding authors**: athervtelkar08@gmail.com (AT) | ameytelkar08@gmail.com (AmT)

---

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

# Software Defect Prediction using SPAM-XAI

A Python implementation of the SPAM-XAI workflow for predicting defect-prone software modules, with a LIME explanation for individual predictions.

Based on: Mustaqeem et al., *A trustworthy hybrid model for transparent software defect prediction: SPAM-XAI*, PLOS ONE 19(7), 2024. DOI: [10.1371/journal.pone.0307112](https://doi.org/10.1371/journal.pone.0307112)

> Mini project (CSE397), School of Computing, SASTRA Deemed to be University. This reproduces the central workflow of the paper; it does not claim to reproduce all published results.

## Pipeline
1. Load a NASA PROMISE dataset (CM1, PC1 or PC2)
2. Standardize features (`StandardScaler`)
3. Balance classes with SMOTE
4. 70/30 stratified train-test split
5. PCA keeping 95% of explained variance
6. MLP classifier (hidden layers 100 and 50, ReLU, Adam, alpha 0.001, max 500 iterations)
7. Evaluate: confusion matrix, classification report, ROC-AUC
8. Explain one test prediction with LIME and save it as an HTML file

## Tech stack
Python, scikit-learn, imbalanced-learn, pandas, NumPy, LIME

## Project structure
```
├── main.py            # full pipeline for CM1, PC1, PC2
├── requirements.txt
├── data/              # cm1.csv, pc1.csv, pc2.csv (NASA PROMISE)
└── docs/              # project report
```

## Usage
```bash
git clone https://github.com/AkashRaj-2004/software-defect-prediction-spam-xai.git
cd software-defect-prediction-spam-xai
pip install -r requirements.txt
python main.py
```
The last column of each CSV must be the defect label. The script prints metrics for each dataset and saves `lime_cm1_explanation.html`, `lime_pc1_explanation.html` and `lime_pc2_explanation.html`.

## Results
| Dataset | Accuracy | F1-score | ROC-AUC |
|---------|----------|----------|---------|
| CM1 | - | - | - |
| PC1 | - | - | - |
| PC2 | - | - | - |

## Authors
Akash Raj.A  Guide: Dr. Ganesh Karthikeyan V.

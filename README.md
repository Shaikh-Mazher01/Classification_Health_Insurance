# Health Insurance → Vehicle Insurance Cross-Sell Prediction

Ranks health-insurance customers by their likelihood of buying vehicle insurance, so a sales team can contact the best prospects first.

🔗 **Live App:** [View Streamlit Application](https://classificationhealthinsurance-byfahad.streamlit.app/)

## Results (20 % untouched test set)
| Metric | Value |
|---|---|
| Model | LightGBM (tuned), chosen on cross-validated PR-AUC |
| ROC-AUC | 0.858 |
| PR-AUC | 0.369 (no-skill ≈ 0.123) |
| Lift, top 30 % of customers | 2.64× |
| Buyers found in top 30 % | 79.3 % |

At the profit-optimal threshold (0.06, from illustrative cost/value assumptions) the model reaches ~98 % of buyers while contacting ~47 % of customers.

**Key findings:** prior vehicle insurance and past vehicle damage dominate (previously insured customers convert at 0.1 %); response peaks at ages 36–55 and rises with vehicle age; the largest sales channel (152) converts at only 2.9 % vs ~19–20 % for others; tenure and licence carry no signal.

**Methodology note:** the notebook (section 6) shows that over-sampling *before* the train/test split inflates results (recall 0.995 vs 0.30 for the same Random Forest, with ~50 % of test rows duplicated in training). All reported numbers use the leak-free setup.

## Repository layout
```
Health_Insurance_Cross_Sell.ipynb   full analysis, modelling and evaluation
app.py                              Streamlit scoring app
models/final_pipeline.joblib        saved scikit-learn pipeline (accepts raw columns)
models/meta.json                    threshold, metrics, importance, dropdown options
data/train.csv                      Kaggle "Health Insurance Cross Sell Prediction" training file (not included)
requirements.txt
```

## Run it
```bash
pip install -r requirements.txt
# put the CSV at data/train.csv  (the notebook also downloads it automatically if missing)
jupyter nbconvert --to notebook --execute Health_Insurance_Cross_Sell.ipynb   # optional: retrains and rewrites models/
streamlit run app.py
```
The notebook takes roughly 10–20 minutes on a laptop CPU. The included `models/` files let you run the app without retraining.

## App features
- **Score one customer** – probability, percentile rank, contact/skip recommendation, expected profit, and a what-if table.
- **Score a customer list** – upload a CSV (template provided), get a ranked list and download it.
- **About the model** – test metrics, feature importance, lift by decile.
- Sidebar lets you change the cost per contact and value per sale; the threshold follows.

## Caveats
Cost and value figures are assumptions. The data is one snapshot (no dates), region/channel codes are anonymised, and gender is a model input – check fairness requirements before any production use.

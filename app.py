"""
Health Insurance -> Vehicle Insurance cross-sell scoring app.

Run:   streamlit run app.py
Needs: models/final_pipeline.joblib and models/meta.json (created by the notebook, section 11).
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_pipeline.joblib"
META_PATH = BASE_DIR / "models" / "meta.json"

FEATURES = ["Gender", "Age", "Driving_License", "Region_Code", "Previously_Insured",
            "Vehicle_Age", "Vehicle_Damage", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
VEHICLE_AGES = ["< 1 Year", "1-2 Year", "> 2 Years"]

st.set_page_config(page_title="Vehicle Insurance Cross-Sell Scorer", page_icon="🚗", layout="wide")


# ───────────────────────── loading ─────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Run the notebook first (section 11 saves `models/final_pipeline.joblib` "
             "and `models/meta.json`), then restart the app.")
    st.stop()

model, meta = load_artifacts()
quantiles = np.array(meta["score_quantiles"])


# ───────────────────────── helpers ─────────────────────────
def score(frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURES])[:, 1]


def percentile_of(p: float) -> float:
    """Share of held-out customers who score below p (0-100)."""
    return float(np.searchsorted(quantiles, p, side="right") - 1)


def validate_batch(df: pd.DataFrame):
    """Return (clean_df, list_of_problems). Rows with problems are dropped and reported."""
    problems = []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {', '.join(missing)}"]
    out = df.copy()
    for c in ["Age", "Driving_License", "Region_Code", "Previously_Insured",
              "Annual_Premium", "Policy_Sales_Channel", "Vintage"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = (
        ~out["Gender"].isin(["Male", "Female"])
        | ~out["Vehicle_Age"].isin(VEHICLE_AGES)
        | ~out["Vehicle_Damage"].isin(["Yes", "No"])
        | out[FEATURES].isna().any(axis=1)
        | ~out["Previously_Insured"].isin([0, 1])
        | ~out["Driving_License"].isin([0, 1])
    )
    if bad.any():
        problems.append(f"{int(bad.sum()):,} row(s) with missing/invalid values were skipped "
                        f"(check Gender, Vehicle_Age, Vehicle_Damage and the 0/1 columns).")
    return out.loc[~bad], problems


# ───────────────────────── sidebar ─────────────────────────
with st.sidebar:
    st.header("Campaign economics")
    st.caption("Illustrative defaults come from the notebook. Replace them with your real figures.")
    cost = st.number_input("Cost of contacting one customer", min_value=0.01,
                           value=float(meta["cost_per_contact"]), step=0.5)
    value = st.number_input("Value of one converted sale", min_value=0.01,
                            value=float(meta["value_per_conversion"]), step=1.0)
    breakeven = cost / value
    st.metric("Break-even probability", f"{breakeven:.1%}")
    use_nb_thr = st.toggle("Use notebook's tuned threshold", value=True,
                           help=f"Notebook threshold = {meta['threshold']:.3f} (chosen from out-of-fold profit).")
    threshold = meta["threshold"] if use_nb_thr else breakeven
    st.write(f"**Active threshold:** {threshold:.3f}")
    st.divider()
    st.caption(f"Model: **{meta['model_name']}**  \nTrained on {meta['n_customers']:,} customers  \n"
               f"Base purchase rate: {meta['base_rate']:.1%}")

st.title("🚗 Vehicle Insurance Cross-Sell Scorer")
st.write("Estimate how likely a **health-insurance customer** is to buy **vehicle insurance**, and decide whether "
         "they are worth contacting.")

tab_single, tab_batch, tab_model = st.tabs(["Score one customer", "Score a customer list", "About the model"])

# ───────────────────────── tab 1: single customer ─────────────────────────
with tab_single:
    regions = [int(r["code"]) for r in meta["regions"]]
    channels = [int(c["code"]) for c in meta["channels"]]
    ranges = meta["ranges"]

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Customer details")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        age = c2.slider("Age", ranges["Age"][0], ranges["Age"][1], 38)
        prev_ins = c1.radio("Already has vehicle insurance?", ["No", "Yes"], horizontal=True)
        damage = c2.radio("Vehicle damaged in the past?", ["Yes", "No"], horizontal=True)
        veh_age = c1.selectbox("Vehicle age", VEHICLE_AGES, index=1)
        license_ = c2.radio("Holds a driving licence?", ["Yes", "No"], horizontal=True)
        region = c1.selectbox("Region code", regions, index=regions.index(28) if 28 in regions else 0,
                              help="Sorted by number of customers.")
        channel = c2.selectbox("Sales channel", channels, index=channels.index(26) if 26 in channels else 0,
                               help="Sorted by number of customers.")
        premium = c1.number_input("Annual premium", min_value=float(ranges["Annual_Premium"][0]),
                                  max_value=float(ranges["Annual_Premium"][1]), value=30000.0, step=500.0)
        vintage = c2.slider("Days as a customer", ranges["Vintage"][0], ranges["Vintage"][1], 150)

    customer = pd.DataFrame([{
        "Gender": gender, "Age": age, "Driving_License": int(license_ == "Yes"), "Region_Code": float(region),
        "Previously_Insured": int(prev_ins == "Yes"), "Vehicle_Age": veh_age, "Vehicle_Damage": damage,
        "Annual_Premium": float(premium), "Policy_Sales_Channel": float(channel), "Vintage": int(vintage),
    }])
    prob = float(score(customer)[0])
    pct = percentile_of(prob)
    expected_profit = prob * value - cost
    contact = prob >= threshold

    with right:
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Purchase probability", f"{prob:.1%}", delta=f"{prob / meta['base_rate']:.1f}× average",
                  delta_color="off")
        m2.metric("Ranks above", f"{pct:.0f}% of customers")
        st.progress(min(max(prob, 0.0), 1.0))
        if contact:
            st.success(f"**Contact this customer.** Score is above the {threshold:.1%} threshold. "
                       f"Expected profit per contact ≈ {expected_profit:+.2f}.")
        else:
            st.warning(f"**Low priority.** Score is below the {threshold:.1%} threshold. "
                       f"Expected profit per contact ≈ {expected_profit:+.2f}.")

        st.markdown("**What would change the score?** (one change at a time)")
        scenarios = []

        def what_if(label, **changes):
            alt = customer.copy()
            for k, v in changes.items():
                alt.loc[0, k] = v
            p_alt = float(score(alt)[0])
            scenarios.append({"Scenario": label, "Probability": p_alt, "Change vs now": p_alt - prob})

        what_if("Already insured" if prev_ins == "No" else "Not insured yet",
                Previously_Insured=1 - customer.loc[0, "Previously_Insured"])
        what_if("No vehicle damage" if damage == "Yes" else "Has vehicle damage",
                Vehicle_Damage="No" if damage == "Yes" else "Yes")
        for v in VEHICLE_AGES:
            if v != veh_age:
                what_if(f"Vehicle age {v}", Vehicle_Age=v)
        what_if("10 years older", Age=min(age + 10, ranges["Age"][1]))
        what_if("10 years younger", Age=max(age - 10, ranges["Age"][0]))
        wi = pd.DataFrame(scenarios)
        st.dataframe(wi.style.format({"Probability": "{:.1%}", "Change vs now": "{:+.1%}"}),
                     hide_index=True)
        st.caption("Region and sales channel also matter: in the training data, regional response rates range "
                   "from about 4% to 19% and channel rates from about 2% to 27%.")

# ───────────────────────── tab 2: batch ─────────────────────────
with tab_batch:
    st.subheader("Score a customer list")
    st.write("Upload a CSV with these columns (an `id` column is optional and is kept):")
    st.code(", ".join(["id"] + FEATURES), language="text")

    template = pd.DataFrame([
        [1, "Male", 44, 1, 28, 0, "> 2 Years", "Yes", 40454, 26, 217],
        [2, "Female", 24, 1, 3, 1, "< 1 Year", "No", 28000, 152, 120],
        [3, "Male", 47, 1, 11, 0, "1-2 Year", "Yes", 35000, 124, 30],
    ], columns=["id"] + FEATURES)
    st.download_button("Download a template CSV", template.to_csv(index=False).encode(), "template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
            raw = None
        if raw is not None:
            if len(raw) > 500_000:
                st.error("Please upload at most 500,000 rows at a time.")
            else:
                clean, problems = validate_batch(raw)
                for msg in problems:
                    st.warning(msg)
                if clean is not None and len(clean):
                    clean = clean.copy()
                    clean["score"] = score(clean)
                    clean["contact"] = clean["score"] >= threshold
                    clean["expected_profit"] = clean["score"] * value - cost
                    clean = clean.sort_values("score", ascending=False)

                    a, b, c = st.columns(3)
                    a.metric("Customers scored", f"{len(clean):,}")
                    b.metric("Recommended to contact", f"{int(clean['contact'].sum()):,}",
                             delta=f"{clean['contact'].mean():.1%} of list", delta_color="off")
                    c.metric("Expected profit (contact list)",
                             f"{clean.loc[clean['contact'], 'expected_profit'].sum():,.0f}")

                    top_pct = st.slider("Or contact only the top X% by score", 1, 100, 30)
                    k = max(1, int(len(clean) * top_pct / 100))
                    st.caption(f"Top {top_pct}% = {k:,} customers, minimum score {clean['score'].iloc[k - 1]:.1%}, "
                               f"expected buyers ≈ {clean['score'].iloc[:k].sum():,.0f}.")

                    show = clean.copy()
                    show["score"] = show["score"].map("{:.1%}".format)
                    st.dataframe(show.head(200), hide_index=True)
                    st.caption("Showing the 200 highest-scoring rows. The download contains everything.")
                    buf = io.StringIO()
                    clean.to_csv(buf, index=False)
                    st.download_button("Download scored list", buf.getvalue().encode(), "scored_customers.csv", "text/csv")
                elif clean is not None:
                    st.error("No valid rows to score.")

# ───────────────────────── tab 3: about ─────────────────────────
with tab_model:
    st.subheader("How well does it work?")
    m = meta["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test ROC-AUC", f"{m['test_roc_auc']:.3f}")
    k2.metric("Test PR-AUC", f"{m['test_pr_auc']:.3f}", help=f"A random model scores about {meta['base_rate']:.3f}.")
    k3.metric("Lift, top 30%", f"{m['test_lift_top30']:.2f}×")
    k4.metric("Buyers found in top 30%", f"{m['test_recall_top30']:.0%}")
    st.caption("All figures are from a 20% held-out test set that was not used for training, tuning or model selection.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**What drives the score** (permutation importance)")
        imp = pd.Series(meta["feature_importance"]).sort_values()
        st.bar_chart(imp, horizontal=True)
    with col_b:
        st.markdown("**Lift by score decile** (1 = highest scored 10%)")
        dec = pd.DataFrame(meta["decile_table"])
        st.bar_chart(dec.set_index("decile")["lift"])

    with st.expander("Decile table"):
        d = pd.DataFrame(meta["decile_table"])
        st.dataframe(d.style.format({"response_rate": "{:.1%}", "avg_score": "{:.1%}", "lift": "{:.2f}",
                                     "cum_buyers_captured": "{:.1%}"}), hide_index=True)
    with st.expander("Model comparison from the notebook"):
        st.dataframe(pd.DataFrame(meta["model_comparison"]), hide_index=True)

    st.markdown("""
**Things to keep in mind**
- The score ranks *likelihood to buy*; it does not prove that contacting someone changes their decision.
- Cost and value inputs in the sidebar are assumptions. The recommendation is only as good as those numbers.
- Region and channel codes are anonymised, so the app cannot explain *why* they matter.
- The data is a single snapshot. Re-check performance on fresh data before relying on it in production.
- Gender is a model input. Confirm that using it complies with your fairness and regulatory requirements.
""")

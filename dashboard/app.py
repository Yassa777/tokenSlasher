import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TokenSlasher Dashboard", layout="wide")

results_dir = st.sidebar.text_input("Results directory", "results")
err_path = Path(results_dir) / "errors.jsonl"
metrics_path = Path(results_dir) / "questions_metrics.txt"

st.title("📊 TokenSlasher Evaluation Dashboard")

if metrics_path.exists():
    st.subheader("Summary Metrics")
    st.text(metrics_path.read_text())
else:
    st.info("Run eval_questions.py first to generate metrics.")

if err_path.exists():
    data = [json.loads(l) for l in err_path.read_text().splitlines()]
    df = pd.DataFrame(data)
    st.subheader("Top Errors")
    st.dataframe(df)
else:
    st.info("No errors.jsonl found.") 
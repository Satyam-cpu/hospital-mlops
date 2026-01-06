import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "inpatientCharges.csv")
PROD_DATA_PATH = os.path.join(BASE_DIR, "logs", "predictions.csv")
REPORT_PATH = os.path.join(BASE_DIR, "monitoring", "drift_report.html")

# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_csv(TRAIN_DATA_PATH)
prod_df = pd.read_csv(PROD_DATA_PATH)

# Clean column names
train_df.columns = train_df.columns.str.strip()

# -------------------------
# OPTIONAL BUT RECOMMENDED
# Fix data-type mismatch (string -> float)
# -------------------------
train_df["Average Covered Charges"] = (
    train_df["Average Covered Charges"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# -------------------------
# SELECT SAME FEATURES
# -------------------------
features = [
    "Provider State",
    "DRG Definition",
    "Total Discharges",
    "Average Covered Charges",
]

train_df = train_df[features]
prod_df = prod_df[features]

# -------------------------
# DRIFT REPORT
# -------------------------
report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)

report.run(
    reference_data=train_df,
    current_data=prod_df,
)

report.save_html(REPORT_PATH)

print("✅ Drift report generated at:", REPORT_PATH)

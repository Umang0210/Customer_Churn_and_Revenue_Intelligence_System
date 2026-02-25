import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")

print("Rows:", len(df))
print("HIGH count:", (df["risk_bucket"] == "HIGH").sum())
print("Total Revenue at Risk:", df["expected_revenue_loss"].sum())
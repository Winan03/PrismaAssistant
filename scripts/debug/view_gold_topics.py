import pandas as pd

df = pd.read_csv("evaluation/gold_standard.csv")
print("Total rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 3 rows:")
print(df[["title", "year", "doi"]].head(3))

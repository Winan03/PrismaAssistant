import pandas as pd
import os

print("Current working dir:", os.getcwd())
print("Files in current dir:", os.listdir("."))
if os.path.exists("logs"):
    print("Files in logs:", os.listdir("logs"))

csv_path = "logs/20260518_052630_1779099990_log_3_FINAL_70percent.csv"
if not os.path.exists(csv_path):
    # Try absolute path
    csv_path = os.path.abspath(csv_path)
    print("Trying absolute path:", csv_path)
    if not os.path.exists(csv_path):
        print("CSV not found at absolute path too!")
        exit(1)

df = pd.read_csv(csv_path)
print("--- DATAFRAME INFO ---")
print(df.info())
print("\n--- SHAPE ---")
print(df.shape)
print("\n--- COLUMNS ---")
print(df.columns.tolist())

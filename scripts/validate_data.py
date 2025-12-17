import pandas as pd

df = pd.read_csv("data/data.csv")

assert df.isnull().sum().sum() == 0, "Dataset contains null values"
assert "Disease" in df.columns, "Missing target column"

import pandas as pd

df = pd.read_csv("data/data.csv")

assert df.isnull().sum().sum() == 0, "Dataset contains null values"
assert "Disease" in df.columns, "Missing target column"

assert len(df) > 100, "Dataset too small"
assert df['Disease'].nunique() > 5, "Not enough disease classes"
print(f"✓ Data validation passed: {len(df)} samples, {df['Disease'].nunique()} diseases")
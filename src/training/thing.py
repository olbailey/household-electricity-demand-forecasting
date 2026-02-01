import pandas as pd

from datasets.sliding_window import convert_sliding_window

df = pd.read_parquet("data/processed/processed_data.parquet")

print(convert_sliding_window(df, 5))
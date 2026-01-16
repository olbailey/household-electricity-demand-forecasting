import pandas as pd
import numpy as np

# There are 25979 rows of incomplete
def find_NANS():
    df_null = df.isnull()
    print(df_null.sum())

    df_null_rows = df[df_null["Voltage"] == True]
    null_dates = df_null_rows["Date"].unique()
    
    print(null_dates)
    print(f"number of days containg null fields: {null_dates.size}")

df = pd.read_csv("data/raw/household_power_consumption.txt", sep=';')
df.replace('?', np.nan, inplace=True)

convert_dict = {header: float for header in list(df)[2:]}
df = df.astype(convert_dict)

df["datetime"] = pd.to_datetime(
    df[["Date", "Time"]].agg(' '.join, axis=1),
    dayfirst=True
)
df.drop(columns=["Date", "Time"], inplace=True)
df = df.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]

print(df.head())

# find_NANS()

import pandas as pd
import numpy as np

# There are 25979 rows of incomplete
def find_NANS():
    df_null = df.isnull()
    print(df_null.sum())

    df_null_rows = df[df_null["Voltage"] == True]
    null_dates = df_null_rows["Datetime"].unique()
    
    print(null_dates)
    print(f"number of days containg null fields: {null_dates.size}")


if __name__ == "__main__":
    df = pd.read_csv("data/raw/household_power_consumption.csv", dtype=object)


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

def convert_datetime(df_: pd.DataFrame) -> pd.DataFrame:
    df_["datetime"] = pd.to_datetime(
        df_[["Date", "Time"]].agg(' '.join, axis=1),
        dayfirst=True
    )
    df_.drop(columns=["Date", "Time"], inplace=True)
    return df_.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]

def correcting_dataframe(df_: pd.DataFrame) -> pd.DataFrame:
    '''
    Replaces figurative nan '?' with numpy.nan values. 
    Corrects data types of columns to floats where necessary
    '''
    df_.replace('?', np.nan, inplace=True)

    convert_dict = {header: float for header in list(df_)[2:]}
    return df_.astype(convert_dict)

df = pd.read_csv("data/raw/household_power_consumption.txt", sep=';')

df = correcting_dataframe(df)
df = convert_datetime(df)

print(df.head())

# find_NANS()

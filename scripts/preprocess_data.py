import os

import numpy as np
import pandas as pd

def convert_datetime(df_: pd.DataFrame) -> pd.DataFrame:
    df_["Datetime"] = pd.to_datetime(
        df_[["Date", "Time"]].agg(' '.join, axis=1),
        dayfirst=True
    )
    df_.drop(columns=["Date", "Time"], inplace=True)
    df_["Datetime"] = pd.to_datetime(df_["Datetime"])
    return df_.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]

def correcting_dataframe(df_: pd.DataFrame) -> pd.DataFrame:
    '''
    Replaces figurative nan '?' with numpy.nan values. 
    Corrects data types of columns to floats where necessary
    '''
    df_.replace('?', np.nan, inplace=True)

    convert_dict = {header: float for header in list(df_)[2:]}
    return df_.astype(convert_dict)

data_dir_path = os.getcwd() + "/data"
data_dir_list = os.listdir(data_dir_path)

if "processed" not in data_dir_list:
    os.mkdir(data_dir_path + "/processed")
    print("Created Directory 'data/processed'")
else:
    print("Directory 'data/processed' already found")

if "processed_data.parquet" not in os.listdir(data_dir_path + "/processed"):
    if "household_power_consumption.csv" not in os.listdir(data_dir_path + "/raw"):
        print("Error! base data set not found!")
        print("Run download_data.py first to download dataset as expected")
    
    else:
        df = pd.read_csv("data/raw/household_power_consumption.csv", dtype=object)
        print("Raw data read in...")

        df = correcting_dataframe(df)
        df = convert_datetime(df)
        print(df.dtypes)
        
        print("Data cleaned.")

        df.to_parquet("data/processed/processed_data.parquet")
        print("Processed data saved to 'data/processed' as processed_data.csv")
else:
    print("Dataset already found in directory")

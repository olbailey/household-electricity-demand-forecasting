import os

import numpy as np
import pandas as pd

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Creates missing data indicator flag, and a count indicating a flowing sum of rows missing data
    To be ran after convert_datetime()
    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # sensor missing valuation
    missing_values_series = df.isna().any(axis=1)
    
    # column to show whether there is any missing data in the row
    df["Missing_mask"] = missing_values_series.astype(int)

    # time since last real reading
    x = np.zeros(len(missing_values_series))

    count = 0
    for i, missing in enumerate(missing_values_series.to_numpy()):
        if missing:
            count += 1
        else:
            count = 0
        x[i] = count

    # column show count since last row that did not have any missing values
    df["Delta_time"] = x

    # replacing nan values with 0
    # zeroing missing data points
    df.fillna(0, inplace=True)

    return df

def convert_datetime(df_: pd.DataFrame) -> pd.DataFrame: # Not currently used as date is subsequently deleted
    """converts the datetime of a dataset to ISO 8601 time format
    To be ran after correcting_dataframe()

    Args:
        df_ (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    df_["Datetime"] = pd.to_datetime(
        df_[["Date", "Time"]].agg(' '.join, axis=1),
        dayfirst=True
    )
    df_.drop(columns=["Date", "Time"], inplace=True)
    df_["Datetime"] = pd.to_datetime(df_["Datetime"])
    return df_.iloc[:, [7, 0, 1, 2, 3, 4, 5, 6]]

def remove_datetime(df_: pd.DataFrame) -> pd.DataFrame:
    """deletes columns regarding the date and time

    Args:
        df_ (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_.drop(columns=["Date", "Time"], inplace=True)
    return df_


def correcting_dataframe(df_: pd.DataFrame) -> pd.DataFrame:
    """Replaces figurative nan '?' with numpy.nan values. 
    Corrects data types of columns to floats where necessary
    To be ran first

    Args:
        df_ (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_.replace('?', np.nan, inplace=True)

    convert_dict = {header: float for header in list(df_)[2:]}
    return df_.astype(convert_dict)

def main():
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
            print("Run scripts/download_data.py first to download dataset as expected")
        
        else:
            df = pd.read_csv("data/raw/household_power_consumption.csv", dtype=object)
            print("Raw data read in...")

            df = correcting_dataframe(df)
            df = remove_datetime(df) # convert_datetime(df)        
            df = handle_missing_data(df)
            print("Data cleaned.")

            df.to_parquet("data/processed/processed_data.parquet")
            print("Processed data saved to 'data/processed' as processed_data.csv")
    else:
        print("Dataset already found in directory")
        answer = input("would you like to delete and reprocess the data? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            os.remove("data/processed/processed_data.parquet")
            main()

if __name__ == "__main__":
    main()

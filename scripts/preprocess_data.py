import os

import numpy as np
import pandas as pd

from utils.clean_data import *

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
            print("Processed data saved to 'data/processed' as processed_data.parquet")
    else:
        print("Dataset already found in directory")
        answer = input("would you like to delete and reprocess the data? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            os.remove("data/processed/processed_data.parquet")
            main()

if __name__ == "__main__":
    main()

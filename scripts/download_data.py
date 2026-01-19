import os

import pandas as pd
from ucimlrepo import fetch_ucirepo 

data_dir_path = os.getcwd() + "/data"
data_dir_list = os.listdir(data_dir_path)
data_dir_list.remove("README.md")

if "raw" not in data_dir_list:
    os.mkdir(data_dir_path + "/raw")
    print("Created Directory 'data/raw'")
else:
    print("Directory 'data/raw' already found")

if "household_power_consumption.csv" not in os.listdir(data_dir_path + "/raw"):
    # fetch dataset 
    individual_household_electric_power_consumption = fetch_ucirepo(id=235)
    print("Dataset fetched")
    
    # data (as pandas dataframes)
    data_df: pd.DataFrame = individual_household_electric_power_consumption.data.features
    
    # Saving data
    data_df.to_csv("data/raw/household_power_consumption.csv", index=False)
    print("Dataset saved to 'data/raw' as .csv")
else:
    print("Dataset already found in directory")
import os

import pandas as pd
# from ucimlrepo import fetch_ucirepo 

import urllib.request
import zipfile

data_dir_path = os.getcwd() + "/data"
data_dir_list = os.listdir(data_dir_path)
data_dir_list.remove("README.md")

if "raw" not in data_dir_list:
    os.mkdir(data_dir_path + "/raw")
    print("Created Directory 'data/raw'")
else:
    print("Directory 'data/raw' already found")

if "household_power_consumption.csv" not in os.listdir(data_dir_path + "/raw"):

    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    urllib.request.urlretrieve(url, "data/raw/household_power_consumption.zip")
    print("Dataset fetched")

    with zipfile.ZipFile("data/raw/household_power_consumption.zip") as z:
        z.extractall("data/raw")
    print("Data extracted")
    
    # data (as pandas dataframes)
    data_df: pd.DataFrame = pd.read_csv("data/raw/household_power_consumption.txt", delimiter=';', low_memory=False)
    
    # Saving data
    data_df.to_csv("data/raw/household_power_consumption.csv", index=False)
    print("Dataset saved to 'data/raw' as .csv")

    os.remove("data/raw/household_power_consumption.zip")
    os.remove("data/raw/household_power_consumption.txt")
    print("Cleaned temporary files")
else:
    print("Dataset already found in directory")
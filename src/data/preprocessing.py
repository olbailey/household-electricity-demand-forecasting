import pandas as pd

def load_raw_data(relative_path: str):
    df = pd.read_csv(relative_path, sep=';')
    print(df.head())

if __name__ == "__main__":
    load_raw_data("data/raw/household_power_consumption.txt")
import pandas as pd
import matplotlib.pyplot as plt

def draw_graph(df: pd.DataFrame, data_label: str, title: str, y_label: str, x_label, length=20, colour="#1f77b4", other_values=[]) -> None:
    plt.figure(figsize=(length, 5))

    if other_values is not None:
        plt.plot(df["Datetime"], df[data_label], label=data_label)
        for item_label in other_values:
            plt.plot(df["Datetime"], df[item_label], label=item_label)
    else:
        plt.plot(df["Datetime"], df[data_label], label=data_label, color=colour)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.grid(True)
    plt.legend()
    plt.show()
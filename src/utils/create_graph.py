import pandas as pd
import matplotlib.pyplot as plt

def draw_graph(df: pd.DataFrame, data_label: str, title: str, y_label: str, length=20) -> None:
    plt.figure(figsize=(length, 5))
    plt.plot(df["DateTime"], df[data_label], label=data_label)

    plt.xlabel("Time of Day")
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.grid(True)
    plt.legend()
    plt.show()
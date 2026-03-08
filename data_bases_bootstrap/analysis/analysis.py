import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def bradford_plot(df):
    df["CumCount"] = df["Frequência"].cumsum()
    df["CumSources"] = np.arange(1, len(df)+1)

    colors = {1: "red", 2: "orange", 3: "blue"}

    plt.scatter(df["CumSources"], df["CumCount"], color="black")
    plt.plot(df["CumSources"], df["CumCount"],  color="black", lw=2, label="Curva cumulativa")

    for zone in [1, 2, 3]:
        subset = df[df["Zone"] == zone]
        plt.fill_between(
                subset["CumSources"],
                0,
                subset["CumCount"],
                color=colors[zone],
                alpha=0.3,
                label=f"Zona {zone}"
                )

    plt.title("Curva de Bradford para bases de dados")
    plt.xlabel("Número cumulativo de bases")
    plt.ylabel("Número cumulativo de menções")
    plt.legend()
    plt.grid(True)

def bradford(df, zones_num):
    df = df.sort_values(by="Frequência", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index+1
    total = df["Frequência"].sum()

    zone_target = total / zones_num

    df["CumFreq"] = df["Frequência"].cumsum()
    zones = []
    current_sum = 0
    current_zone = 1
    zone_limits = []

    for i in range(1, zones_num):
        zone_limits.append(df[df["CumFreq"] >= i*zone_target].index[0])

    df["Zone"] = zones_num
    start = 0
    for zone_number, end_idx in enumerate(zone_limits+[len(df)-1], start=1):
        df.loc[start:end_idx, "Zone"] = zone_number
        start = end_idx+1

    return df

def data_in(file_extension: str) -> dict:
    planilhas = [item for item in os.listdir('./data') if file_extension in item]
    objects = {}
    for planilha in planilhas:
        xlsx = pd.ExcelFile(planilha)
        dict = pd.read_excel(planilha, sheet_name=None)
        objects[planilha] = xlsx
    return objects

def describe(data: dict):
    for item in data:
        sheets = item.sheet_names
        #for sheet in sheets:
            

if __name__ == '__main__':

    df = pd.read_csv('data_base_evaluation - scopus_results.csv')

    print(df.head())

    #plt.scatter(df["Rank"], df["Frequência"], color='blue', label='Frequência Real')
    df = bradford(df, 3)
    bradford_plot(df)

    plt.xscale('log')
    plt.yscale('log')

    plt.show()



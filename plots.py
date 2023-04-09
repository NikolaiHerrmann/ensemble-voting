
import matplotlib.pyplot as plt 
import json
import os
import seaborn as sns
import pandas as pd


def read_data(dataset):
    with open(os.path.join("data", dataset), "r") as f:
        return json.load(f)


def plot(data):
    f1_scores = []
    times = []

    for key, val in data.items():
        comb = key.split(" ")
        rule = comb[1].replace("_", " ")
        
        for i in val[0]:
            f1_scores.append([int(comb[0]), rule, i])

        if comb[0] == "1":
            continue
        for i in val[1]:
            times.append([comb[0], rule, i])

    colors = ["limegreen", "orangered", "orange", "dodgerblue", "teal"]
    baseline = "darkorchid"

    plt.rcParams['figure.figsize'] = (10, 5)

    df = pd.DataFrame(f1_scores, columns=["Ensemble Size", "Voting Rule", "F1 Score"])
    p = sns.boxplot(x = df['Ensemble Size'], y = df['F1 Score'], hue = df['Voting Rule'], palette=[baseline]+colors)
    sns.move_legend(p, "lower left")
    plt.axhline(y = df[df["Ensemble Size"] <= 1].mean(numeric_only=True)[1], color = baseline, linestyle = '--', linewidth=2.2)
    plt.title("Performance of Wine Dataset")
    plt.grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2)
    plt.savefig("performance.pdf", bbox_inches="tight")
    plt.show()

    df2 = pd.DataFrame(times, columns=["Ensemble Size", "Voting Rule", "Prediction Time (ms)"])
    sns.boxplot(x = df2['Ensemble Size'], y = df2['Prediction Time (ms)'], hue = df2['Voting Rule'], palette=colors)
    plt.title("Prediction Time of Wine Dataset")
    plt.grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2)
    plt.savefig("time.pdf", bbox_inches="tight")
    plt.show()

plot(read_data("wine.json"))
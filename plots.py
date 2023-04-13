
import matplotlib.pyplot as plt 
import json
import os
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.patches as mpatches
import scikit_posthocs as sp


def read_data(dataset):
    with open(os.path.join("data", dataset), "r") as f:
        return json.load(f)


def plot(dataset):
    data = read_data(dataset)

    f1_scores = []
    times = []
    title_ = dataset.split("_")[0].title()
    rules = ["Baseline (-- = median)", "Borda", "Copeland", "Sum", "Kemeny", "Plurality", "STV"]

    for key, val in data.items():
        offset = 1 if key != "1" else 0
        
        for i, exp_data in enumerate(np.array(val[0]).T):
            if rules[i + offset] == "Sum":
                continue
            for col in exp_data:
                f1_scores.append([int(key), rules[i + offset], col])

        if offset == 0:
            continue
        for i, exp_data in enumerate(np.array(val[1]).T):
            if rules[i + offset] == "Sum":
                continue
            for col in exp_data:
                times.append([int(key), rules[i + offset], col])

    colors = ["limegreen", "orangered", "orange", "dodgerblue", "cyan"]
    baseline = "darkorchid"

    plt.rcParams['figure.figsize'] = (10, 5)

    df = pd.DataFrame(f1_scores, columns=["Ensemble Size", "Voting Rule", "F1 Score"])
    p = sns.boxplot(x = df['Ensemble Size'], y = df['F1 Score'], hue = df['Voting Rule'], palette=[baseline]+colors)
    sns.move_legend(p, "lower left")
    plt.axhline(y = df[df["Ensemble Size"] <= 1].median(numeric_only=True)[1], color = baseline, linestyle = '--', linewidth=2.2, zorder=-1)
    plt.title("Performance of " + title_ + " Dataset")
    plt.grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2)
    plt.savefig("performance_" + title_ + ".pdf", bbox_inches="tight")
    #plt.show()

    df2 = pd.DataFrame(times, columns=["Ensemble Size", "Voting Rule", "Prediction Time (ms)"])
    p = sns.lmplot(x = 'Ensemble Size', y = 'Prediction Time (ms)', hue = 'Voting Rule', data=df2, palette=colors,scatter=False,legend_out=False)
    plt.title("Prediction Time of " + title_ + " Dataset")
    plt.grid(color='#95a5a6', linestyle='-', linewidth=0.8, alpha=0.2)
    
    handles = []
    for i, r in enumerate(["Borda", "Copeland", "Kemeny", "Plurality", "STV"]):
        d = df2[df2['Voting Rule'].str.contains(r)]
        slope, inter, rval, pval, stderr = stats.linregress(d["Ensemble Size"], d["Prediction Time (ms)"])
        add_ = "-" if inter < 0 else "+"
        l = f"  $R^2=${round(rval, 3)}  ($y=${round(slope, 3)}$x {add_}${abs(round(inter, 3))})"
        handles.append(mpatches.Patch(color=colors[i], label=r"$\bf{" + str(r) + "}$" + l))

    plt.legend(handles=handles, prop={'size': 8})
    plt.savefig("time_" + title_ + ".pdf", bbox_inches="tight")
    #plt.show()

    filter = df[df["Ensemble Size"] == 60]
    lists = []
    rules = ["Borda", "Copeland", "Kemeny", "Plurality", "STV"]
    means = []
    for name in rules:
        arr = np.array(filter[filter["Voting Rule"] == name]["F1 Score"])
        print(name, np.mean(arr))
        print(arr)
        means.append(np.mean(arr))
        lists.append(arr)
    print(stats.friedmanchisquare(*lists))
    print(sp.posthoc_nemenyi_friedman(np.array(lists).T).round(3))
    print(rules)
    print(np.round(means, 4))
    

plot("wine_new.json")
plot("dermatology_new.json")
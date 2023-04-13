
import matplotlib.pyplot as plt
import numpy as np

baseline = [0.8599, 0.8652, 0.8626, 0.8630, 0.8633, 0.8626]

ensembles = [0.8886, 0.8966,  0.9008, 0.9043, 0.9066, 0.9068]

sizes = ["5", "7", "10", "20", "40", "50"]

rules = ["Kemeny", "Kemeny/\n Plurality", "Kemeny", "Plurality", "Plurality", "Plurality"]

m = np.mean(baseline)

plt.bar(sizes, ensembles, color="purple")
plt.axhline(m, color="red", linestyle="--", label="Baseline (average single classifier = " + str(round(m, 3)) + ")", linewidth=2)
plt.xlabel("Ensemble Size")
plt.ylabel("Average F1-score")
plt.ylim(0.85, 0.92)
plt.title("Summarizing Results of Experiment 1")
plt.legend()

xlocs, _ = plt.xticks()
for i, v in enumerate(rules):
    plt.text(xlocs[i] - 0.4, ensembles[i] + 0.001, v)
    plt.text(xlocs[i] - 0.25, ensembles[i] - 0.005, str(round(ensembles[i], 3)), color="white")

plt.savefig("exp1.pdf", bbox_inches="tight")
plt.show()

datasets = ["Anneal", "Autos", "Balance", "Cars", "Dermatology",
         "Glass", "Iris", "Lymphography", "Vowel",
         "Wine", "Zoo"]


avgs = [0.9365, 0.9413, 0.9396, 0.9402, 0.9416, 0.8720, 0.9177]
names = ["VORACE + Borda", "VORACE + Plurality", "VORACE + Copeland", "VORACE + Kemeny", "VORACE + Sum", "RF", "XGBoost"]

avgs_2 = [0.8724, 0.8574, 0.8666, 0.8636]
names_2 = ["VORACE + Majority", "VORACE + Sum", "RF", "XGBoost"]

plt.cla()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[1].bar(names, avgs, color="purple")
ax[1].set_ylim(0.86, 0.95)
ax[1].set_xlabel("Ensemble Method")
ax[1].set_ylabel("Average F1-score")
ax[1].set_title("Multi-class Classification")

ax[0].bar(names_2, avgs_2, color="purple")
ax[0].set_ylim(0.85, 0.88)
ax[0].set_xlabel("Ensemble Method")
ax[0].set_ylabel("Average F1-score")
ax[0].set_title("Binary Classification")

plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)

xlocs = ax[1].get_xticks()
for i, v in enumerate(avgs):
    ax[1].text(xlocs[i] - 0.3, v - 0.005, str(round(v, 3)), color="white")

xlocs = ax[0].get_xticks()
for i, v in enumerate(avgs_2):
    ax[0].text(xlocs[i] - 0.2, v - 0.002, str(round(v, 3)), color="white")

plt.suptitle("Summarizing Results of Experiment 2")
plt.savefig("exp2.pdf", bbox_inches="tight")
plt.show()
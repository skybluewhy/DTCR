import matplotlib
matplotlib.use('TkAgg')

res = {"DTCR": {"PISTE": 0.9053070472887768, "ERGO": 0.8998671336696091, "epiTCR": 0.8957685861671981},
       "DTCR-R": {"PISTE": 0.8935906661911014, "ERGO": 0.8856537876874905, "epiTCR": 0.89},
       "DTCR-M": {"PISTE": 0.8672425462500001, "ERGO": 0.8625915703125, "epiTCR": 0.8360134300000001},
       "TCR-TRANSLATE": {"PISTE": 0.8592286250000001, "ERGO": 0.8683741250000001, "epiTCR": 0.8349126874999999},
       "Real Binding TCRs": {"PISTE": 0.9053732967523462, "ERGO": 0.8809554879983952, "epiTCR": 0.8769955826460573},
}

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(res).T

plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

ax = df.plot(kind='bar', width=0.8, edgecolor='black')

plt.title('', fontsize=15, fontweight='bold')
plt.xlabel('Models', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.legend(title='Prediction Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')

plt.ylim(0.80, 0.95)

for i in ax.containers:
    ax.bar_label(i,
                 fmt='%.3f',
                 padding=2,
                 fontsize=8,
                 rotation=90)

plt.tight_layout()
plt.savefig("comparsion_unipep.svg")

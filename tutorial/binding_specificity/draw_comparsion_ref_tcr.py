import matplotlib
matplotlib.use('TkAgg')

res = {"DTCR": {"PISTE": 0.9264718981831416, "TEIM": 0.9012353342571597, "ERGO": 0.934445901875,
                "epiTCR": 0.8943023353125001, "NetTCR-2.0": 0.9247589841503963},
       "DTCR-R": {"PISTE": 0.8829555691267463, "TEIM": 0.8658527947723582, "ERGO": 0.9136777040624999,
                "epiTCR": 0.8264279587499999, "NetTCR-2.0": 0.8986450273118172},
       "DTCR-M": {"PISTE": 0.8596637752757352, "TEIM": 0.8287780985405526, "ERGO": 0.8756773715625,
                "epiTCR": 0.8026099265625001, "NetTCR-2.0": 0.9306123421875},
       "TCR-TRANSLATE": {"PISTE": 0.8544629769497594, "TEIM": 0.7627481257692738, "ERGO": 0.8824121875,
                "epiTCR": 0.8721601760281076, "NetTCR-2.0": 0.8738427499999999},
       "GRATCR": {"PISTE": 0.8501371007357502, "TEIM": 0.685913787988175, "ERGO": 0.8677739771874999,
                "epiTCR": 0.8601286128124999, "NetTCR-2.0": 0.8549902084375001},
       "Real Binding TCRs": {"PISTE": 0.9377029219141446, "TEIM": 0.900787721391306, "ERGO": 0.9387555998291809,
                "epiTCR": 0.9186582696110313, "NetTCR-2.0": 0.9328070838497262},
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

plt.ylim(0.65, 1.0)

for i in ax.containers:
    ax.bar_label(i,
                 fmt='%.3f',
                 padding=2,
                 fontsize=8,
                 rotation=90)

plt.tight_layout()
plt.savefig("comparsion_ref_tcr.svg")

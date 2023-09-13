import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


DIR = 'files/performance/'
files = [os.path.join(DIR, f) for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f))]

for f in files:
    print(f)
    df = pd.read_csv(f, index_col=0)
    sns.barplot(data=df, x='AUC', y='Phylum Name', hue='Shuffled', capsize=0.2, orient="h")
    plt.savefig(f.partition('.csv')[0].replace('files', 'figures')+'.png')
    plt.close()

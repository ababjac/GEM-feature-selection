import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


DIR = 'files/performance_5/'
files = [os.path.join(DIR, f) for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f))]

for f in files:
    print(f)

    data, _, typ = f.split('-')
    data = data.split('/')[-1]
    typ = typ.partition('.')[0]
    df = pd.read_csv(f, index_col=0)
    df = df.sort_values(by='Phylum Name')
    sns.barplot(data=df, x='AUC', y='Phylum Name', hue='Shuffled', capsize=0.2, orient="h")
    plt.title('AUC Performance - '+data+' '+typ)
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f.partition('.csv')[0].replace('files', 'figures')+'.png')
    plt.close()

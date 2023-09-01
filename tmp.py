import pandas as pd

DATA = 'TBG'
SHUF = 'actual'
TYPE = 'pathway'
CUTOFF = 'cutoff_0.95'

df = pd.read_csv('./files/{}/{}/{}-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(CUTOFF, DATA, DATA, SHUF, TYPE), index_col=0)
print(set(df['phylum_name']))

DATA = 'GEM'

df = pd.read_csv('./files/{}/{}/{}-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(CUTOFF, DATA, DATA, SHUF, TYPE), index_col=0)
print(set(df['phylum_name']))

DATA = 'HG'

df = pd.read_csv('./files/{}/{}/{}-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(CUTOFF, DATA, DATA, SHUF, TYPE), index_col=0)
print(set(df['phylum_name']))

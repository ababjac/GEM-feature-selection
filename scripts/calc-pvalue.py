import pandas as pd
import numpy as np
import scipy.stats as st
import os

CUTOFF = ['cutoff_0.90', 'cutoff_0.95']
DATA = ['TBG', 'GEM', 'HG']
SHUF = ['actual', 'shuffled']
TYPE = ['pathway', 'annotation']

for cutoff in CUTOFF:
    for data in DATA:
        PATH = os.getcwd() + '/files/{}/{}/'.format(cutoff, data)

        for shuf in SHUF:
            for typ in TYPE:
                df = pd.read_csv(PATH+'{}-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(data, shuf, typ), index_col=0)

                t = []
                p = []
                n = 50 #for how many trees were built in bagging regressor
                m = 0.0 #the true mean of 0 meaning the coefficient is not important
                for sm, sv in zip(df['coef'], df['coef_sd']):
                    tt = (sm-m)/(sv/np.sqrt(float(n)))  # t-statistic for mean
                    pval = st.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)

                    t.append(tt)
                    p.append(pval)

                df['t-statistic'] = t
                df['p-value'] = p

                df.to_csv(PATH+'{}-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(data, shuf, typ))

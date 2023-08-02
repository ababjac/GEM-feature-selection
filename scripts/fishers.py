from scipy.stats import fisher_exact, boschloo_exact
from scipy.stats.contingency import chi2_contingency
import numpy as np
import pandas as pd

DATA = 'GEM'
TYPE = 'pathway'


s_df = pd.read_csv('./files/{}/{}-actual-bootstrapped-{}_feature_counts_by_phylum_significant.csv'.format(DATA, DATA, TYPE), index_col=0)
i_df = pd.read_csv('./files/{}/{}-actual-bootstrapped-{}_feature_counts_by_phylum_insignificant.csv'.format(DATA, DATA, TYPE), index_col=0)

#print(s_df)
#print(i_df)

s_l = [0,0]
i_l = [0,0]

for name in s_df['feature_name']:
    #print(s_df.loc[s_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0]) 
    #print(s_df.loc[s_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0])
    if name.__contains__('biosynthesis'):
        s_l[0] += s_df.loc[s_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0] + s_df.loc[s_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0]
        i_l[0] += i_df.loc[i_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0] + i_df.loc[i_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0]

    else:
        s_l[1] += s_df.loc[s_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0] + s_df.loc[s_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0]
        i_l[1] += i_df.loc[i_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0] + i_df.loc[i_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0]

cont_table = np.array([s_l, i_l]).reshape(2,2)
print(cont_table)

print(fisher_exact(cont_table))

#c_l = [0,0]
#u_l = [0,0]

#for name in s_df['feature_name']:
#    if name.__contains__('biosynthesis'):
#        c_l[0] += s_df.loc[s_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0]
#        u_l[0] += s_df.loc[s_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0]

#    else:
#        c_l[1] += s_df.loc[s_df['feature_name'] == name, 'cultured_feature_counts'].to_numpy()[0]
#        u_l[1] += s_df.loc[s_df['feature_name'] == name, 'uncultured_feature_counts'].to_numpy()[0]

#cont_table = np.array([c_l, u_l]).reshape(2,2)
#print(cont_table)

#print(fisher_exact(cont_table))


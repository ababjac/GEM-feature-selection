import pandas as pd
import numpy as np
import helpers
import os

# df = pd.read_csv('files/LASSO-coefficients-annotation-list.txt', sep=' : ', names=['features', 'coef'])
# df['abs_coef'] = abs(df['coef'])
# df = df[df['abs_coef'] > 0.01]
# df = df.sort_values(by='abs_coef')
#
# l = [colname+' : '+str(coefficient) for colname, coefficient in zip(df['features'], df['coef'])]
# helpers.write_list_to_file('files/LASSO-coefficients-annotation-list-reduced.txt', l)

# DIRECTORY = 'files/by-phylum-annotation/'
#
# df = pd.DataFrame(columns=['phylum_name', 'feature_name', 'coefficient'])
# #count = 0
#
# with os.scandir(DIRECTORY) as d:
#     for entry in d:
#         if entry.name.endswith('.txt') and entry.is_file():
#             path = os.path.join(DIRECTORY, entry.name)
#
#             intext = pd.read_csv(path, sep=' : ', names=['feature_name', 'coefficient'], engine='python')
#             #count += intext.shape[0]
#             if intext.empty:
#                 continue
#
#             pathway_name = entry.name.split('.')[0].split('-')[-1]
#             l = [pathway_name]*intext.shape[0]
#             intext['phylum_name'] = l
#
#             #print(intext)
#             df = df.append(intext)
#             #print(df)
#
# #print(df)
# #print(count)
# df.to_csv('files/by-phylum-lasso-coefficients-annotation-full.csv')

DATA = 'GEM'
SHUF = 'shuffled'
TYPE = 'annotation'
SIG = 'significant'

df = pd.read_csv('./files/{}/{}-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(DATA, DATA, SHUF, TYPE), index_col=0)

if SIG == 'significant':
    df = df[df['significant'] == True]
else:
    df = df[df['significant'] == False]

df['predicted'] = np.where(df['coef'] >=0, 'C', 'U')

phylums = set(df['phylum_name'].tolist())
features = set(df['feature_name'].tolist())

new_df = pd.DataFrame()
df_C = df[df['predicted'] == 'C']
df_U = df[df['predicted'] == 'U']
count_C = [0]*len(features)
count_U = [0]*len(features)
for phylum in phylums:
    sub_C = df_C[df_C['phylum_name'] == phylum]
    sub_U = df_U[df_U['phylum_name'] == phylum]

    for i, feature in zip(list(range(len(features))), features):
        features_C = set(sub_C['feature_name'].tolist())
        features_U = set(sub_U['feature_name'].tolist())

        if feature in features_C:
            count_C[i] += 1

        if feature in features_U:
            count_U[i] += 1

new_df['feature_name'] = list(features)
new_df['cultured_feature_counts'] = count_C
new_df['uncultured_feature_counts'] = count_U
new_df.to_csv('./files/{}/{}-{}-{}_feature_counts_by_phylum_{}.csv'.format(DATA, DATA, SHUF, TYPE, SIG))

# curr_dir = os.getcwd()
# path_file = curr_dir+'/data/GEM_data/pathway_features_counts_wide.tsv'
# annot_file = curr_dir+'/data/GEM_data/annotation_features_counts_wide.tsv'
# path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=helpers.detect_encoding(path_file))
# annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=helpers.detect_encoding(annot_file))
#
# helpers.write_list_to_file(curr_dir+'/files/pathway_feature_names_full.txt', list(path_features.columns))
# helpers.write_list_to_file(curr_dir+'/files/annotation_feature_names_full.txt', list(annot_features.columns))
#
# print(len(path_features.columns), len(annot_features.columns))

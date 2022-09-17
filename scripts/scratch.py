import pandas as pd
import helpers
import os

# df = pd.read_csv('files/LASSO-coefficients-annotation-list.txt', sep=' : ', names=['features', 'coef'])
# df['abs_coef'] = abs(df['coef'])
# df = df[df['abs_coef'] > 0.01]
# df = df.sort_values(by='abs_coef')
#
# l = [colname+' : '+str(coefficient) for colname, coefficient in zip(df['features'], df['coef'])]
# helpers.write_list_to_file('files/LASSO-coefficients-annotation-list-reduced.txt', l)

DIRECTORY = 'files/by-phylum/'

df = pd.DataFrame(columns=['pathway_name', 'feature_name', 'coefficient'])
#count = 0

with os.scandir(DIRECTORY) as d:
    for entry in d:
        if entry.name.endswith('.txt') and entry.is_file():
            path = os.path.join(DIRECTORY, entry.name)

            intext = pd.read_csv(path, sep=' : ', names=['feature_name', 'coefficient'], engine='python')
            #count += intext.shape[0]
            if intext.empty:
                continue

            pathway_name = entry.name.split('.')[0].split('-')[-1]
            l = [pathway_name]*intext.shape[0]
            intext['pathway_name'] = l

            #print(intext)
            df = df.append(intext)
            #print(df)

#print(df)
#print(count)
df.to_csv('files/by-phylum-lasso-coefficients-pathway-full.csv')

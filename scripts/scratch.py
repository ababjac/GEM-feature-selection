import pandas as pd
import helpers

df = pd.read_csv('files/LASSO-coefficients-annotation-list.txt', sep=' : ', names=['features', 'coef'])
df['abs_coef'] = abs(df['coef'])
df = df[df['abs_coef'] > 0.01]
df = df.sort_values(by='abs_coef')

l = [colname+' : '+str(coefficient) for colname, coefficient in zip(df['features'], df['coef'])]
helpers.write_list_to_file('files/LASSO-coefficients-annotation-list-reduced.txt', l)

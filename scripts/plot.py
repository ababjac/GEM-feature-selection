import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import os


def make_graph(textfile, image_save):
    df = pd.read_csv(textfile, sep=' : ', names=['features', 'coef'])

    df['abs_coef'] = abs(df['coef'])
    df['direction'] = np.where(df['coef'] > 0, 'Positive', 'Negative') #1 for positive, 0 for negative
    sorted_df = df.sort_values(by='abs_coef')
    sorted_df['position'] = list(range(sorted_df.shape[0]))

    fig = px.scatter(sorted_df, x='abs_coef', y='position', color='direction')
    fig.update_yaxes(tickvals=sorted_df['position'], ticktext=sorted_df['features'])
    fig.update_layout(
        title_text='GEM: '+image_save.split('.')[0].split('-')[-1],
        title_x=0.65,
        title_y=0.98,
        title_xanchor='center',
        title_yanchor='top',
        xaxis_title='LASSO Coefficient (Absolute) Value',
        yaxis_title='Pathway Features',
        legend=dict(title='Direction', orientation='h', x=0.3, y=0.1),
        autosize=False,
        width=1600,
        height=1200
    )
    #fig.show()
    fig.write_image(image_save)

DIRECTORY = 'files/by-phylum-annotation/'

with os.scandir(DIRECTORY) as d:
    for entry in d:
        if entry.name.endswith('.txt') and entry.is_file():
            path = os.path.join(DIRECTORY, entry.name)

            make_graph(path, 'figures/by-phylum-annotation/'+entry.name.split('.')[0]+'.pdf')

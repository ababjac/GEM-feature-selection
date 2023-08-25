import pandas as pd
import numpy as np
import scipy.stats as st
from statistics import mean, stdev
import os
import helpers

DATA = 'annotation'

if __name__ == '__main__':

    print('Loading data...')
    curr_dir = os.getcwd()
    #f = open(curr_dir+'/files/GEM-by-phylum-{}-uncultured-distibution-test.txt'.format(DATA), 'w+') #open log file

    meta_file = curr_dir+'/data/GEM_data/GEM_metadata.tsv'
    path_file = curr_dir+'/data/GEM_data/pathway_features_counts_wide.tsv'
    annot_file = curr_dir+'/data/GEM_data/annotation_features_counts_wide.tsv'

    metadata = pd.read_csv(meta_file, sep='\t', header=0, encoding=helpers.detect_encoding(meta_file))

    if DATA == 'pathway':
        path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=helpers.detect_encoding(path_file))
        path_features = helpers.normalize_abundances(path_features)
        data = pd.merge(metadata, path_features, on='genome_id', how='inner')
    else:
        annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=helpers.detect_encoding(annot_file))
        annot_features = helpers.normalize_abundances(annot_features)
        data = pd.merge(metadata, annot_features, on='genome_id', how='inner')

    phylum_list = set(list(data['phylum']))

    full_df = pd.DataFrame(columns=['phylum_name', 'feature_name', 'statistic', 'statistic_lower_95', 'statistic_upper_95', 'p-value', 'p-value_lower_95', 'p-value_upper_95', 'significant'])
    for phylum in phylum_list:
        print(phylum)
        if pd.isna(phylum):
            continue

        data1 = data[data['phylum'] == phylum] #uncultured things
        data2 = data[data['cultured.status'] == 'cultured'] #cultured things
        label_strings = data1['cultured.status']

        #skip phyla that are not fully uncultured or have less than 5 samples
        if (sum(label_strings == 'cultured') != 0) or (sum(label_strings == 'uncultured') < 5):
            continue

        features = data1.loc[:, ~data1.columns.isin(['genome_id','cultured.status'])] #remove labels
        features = features.loc[:, ~features.columns.isin(['culture.level',
                                                           'taxonomic.dist',
                                                           'domain',
                                                           'phylum',
                                                           'class',
                                                           'order',
                                                           'family',
                                                           'genus',
                                                           'species',
                                                           'completeness',
                                                           'genome_length'
                                                           ])]

    
        #f.write('Phylum: {}\n'.format(phylum))
        f_stats = []
        f_pvals = []
        ci_stats_upper = []
        ci_stats_lower = []
        ci_pvals_upper = []
        ci_pvals_lower = []
        sigs = []
        for col in features.columns:
            stats = []
            pvals = []

            for i in range(1000):
                A = data1[col]
                B = data2[col].sample(n=len(A), random_state=i)

                stat, pval = st.ttest_ind(A, B, nan_policy='raise', random_state=i)
                stats.append(stat)
                pvals.append(pval)

            conf_it_st = st.t.interval(alpha=0.95, df=len(stats)-1, loc=mean(stats), scale=st.sem(stats))
            conf_it_pv = st.t.interval(alpha=0.95, df=len(pvals)-1, loc=mean(pvals), scale=st.sem(pvals))

            f_stats.append(mean(stats))
            f_pvals.append(mean(pvals))
            ci_stats_upper.append(conf_it_st[1])
            ci_stats_lower.append(conf_it_st[0])
            ci_pvals_upper.append(conf_it_pv[1])
            ci_pvals_lower.append(conf_it_pv[0])
            
            if conf_it_pv[1] <= 0.05:
                sigs.append(True)
            else:
                sigs.append(False)

        df = pd.DataFrame()
        df['phylum_name'] = [phylum]*len(features.columns)
        df['feature_name'] = features.columns
        df['statistic'] = f_stats
        df['statistic_lower_95'] = ci_stats_lower
        df['statistic_upper_95'] = ci_stats_upper
        df['p-value'] = f_pvals
        df['p-value_lower_95'] = ci_pvals_lower
        df['p-value_upper_95'] = ci_pvals_upper
        df['significant'] = sigs

        full_df = full_df.append(df)

    full_df.to_csv(curr_dir+'/files/GEM-by-phylum-{}-uncultured-distibution-test.txt'.format(DATA))
    #f.write('\tFeature: {} (significant = {})\n\t\tStatistic: {} (lower = {}, upper = {})\n\t\tP-Value: {} (lower = {}, upper = {})\n\n'.format(col, sig, mean(stats), conf_it_st[0], conf_it_st[1], mean(pvals), conf_it_pv[0], conf_it_pv[1]))
    #f.close()

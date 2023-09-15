import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import statistics

from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.utils import shuffle
import scipy.stats as st

import helpers

#--------------------------------------------------------------------------------------------------#

def run_LASSO(X_train_scaled, X_test_scaled, y_train, y_test, phylum=None, param_grid=None, random_state=872510):
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state)
    model = BaggingRegressor(base_estimator=lasso, n_estimators=50, bootstrap=True, verbose=0, random_state=random_state)

    model.fit(X_train_scaled, y_train)
    return model
'''
    feature_names = list(set(X_train.columns.tolist()))
    counts = dict.fromkeys(feature_names, 0)
    coefs = dict(zip(feature_names, ([] for _ in feature_names)))

    for m in model.estimators_:
        coefficients = m.coef_
        for f, c in zip(feature_names, coefficients[0]):
            coefs[f].append(c)
            if c != 0:
                counts[f] += 1

    means = dict.fromkeys(feature_names, None)
    std_devs = dict.fromkeys(feature_names, None)

    for k, v in coefs.items():
        means[k] = statistics.mean(v)
        std_devs[k] = statistics.stdev(v)

    l90 = []
    u90 = []
    sig = []
    for data in coefs.values():
         conf_it = st.t.interval(alpha=0.95, df=len(data)-1, loc=statistics.mean(data), scale=st.sem(data))

         if conf_it[0] <= 0 and conf_it[1] >= 0:
             sig.append(False)
         elif np.isnan(conf_it[0]) or np.isnan(conf_it[1]):
             sig.append(False)
         else:
             sig.append(True)

         l90.append(conf_it[0])
         u90.append(conf_it[1])

    imp = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=random_state)

    output = pd.DataFrame()
    output['phylum_name'] = [phylum]*len(feature_names)
    output['feature_name'] = feature_names
    output['coef'] = means.values()
    output['coef_sd'] = std_devs.values()
    output['lower_95'] = l90
    output['upper_95'] = u90
    output['count'] = counts.values()
    output['significant'] = sig
    output['permutation_importance'] = imp.importances_mean
    #print(output)

    remove = output[output['significant'] == False]['feature_name']
    #print(remove)

    if len(remove) < len(X_train_scaled.columns):
       LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
       LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]
    else:
       print('LASSO COULDNT DECIDE')
       LASSO_train = X_train_scaled
       LASSO_test = X_test_scaled

    return LASSO_train, LASSO_test, output, model
'''

#--------------------------------------------------------------------------------------------------#

SHUF = 'shuffled'
DATA = 'annotation'

if __name__ == '__main__':

    print('Loading data...')
    #f = open('./files/cutoff_0.95/HG/HG-{}-by-phylum-{}-LASSO-metrics-SMOTE.log.txt'.format(SHUF, DATA), 'w+') #open log file
    curr_dir = os.getcwd()
    f = open(curr_dir+'/files/benchmarking/EHG-{}-{}.txt'.format(SHUF, DATA), 'w+')

    meta_file = curr_dir+'/data/human_gut_data/COG_counts/early_human_gut_cultured_level_taxa.csv'
    path_file = curr_dir+'/data/human_gut_data/COG_counts/early_human_gut_pathway_count.csv'
    annot_file = curr_dir+'/data/human_gut_data/COG_counts/early_human_gut_annot_count.csv'

    metadata = pd.read_csv(meta_file, sep=',', header=0, index_col=0, encoding=helpers.detect_encoding(meta_file))

    if DATA == 'pathway':
        path_features = pd.read_csv(path_file, sep=',', quotechar='"', index_col=0, header=0, encoding=helpers.detect_encoding(path_file))
        path_features.rename(columns={'filename' : 'Genome'}, inplace=True)
        path_features = helpers.normalize_abundances(path_features)
        data = pd.merge(metadata, path_features, on='Genome', how='inner')

    else:
        annot_features = pd.read_csv(annot_file, sep=',', quotechar='"', index_col=0, header=0, encoding=helpers.detect_encoding(annot_file))
        annot_features.rename(columns={'filename' : 'Genome'}, inplace=True)
        annot_features = helpers.normalize_abundances(annot_features)
        data = pd.merge(metadata, annot_features, on='Genome', how='inner')

    phylum_list = set(list(data['phylum']))

    #print(phylum_list)
    #print(data)
    #full_df = pd.DataFrame(columns=['phylum_name', 'feature_name', 'coef', 'coef_sd', 'lower_95', 'upper_95', 'count', 'significant', 'permutation_importance'])
    for phylum in phylum_list:
        if pd.isna(phylum):
            continue

        data1 = data[data['phylum'] == phylum]

        label_strings = data1['cultured_level']
        print(phylum, ':', data1.shape)
        if data1.shape[0] < 10:
       #     print('Cultured:  ', sum(label_strings == 'species'), 'Uncultured: ', sum(label_strings != 'species'))
       #     print()
     #       print('not enough data')
            continue

        n_neighbors = 3
        
      #  print('Cultured:  ', sum(label_strings == 'species'), 'Uncultured: ', sum(label_strings != 'species'))
      #  print()

        if 'species' not in set(list(label_strings)):
    #        print('Cultured:  0, Uncultured: ', len(label_strings))
     #       print()
            continue

        if (sum(label_strings == 'species') < 5) or (sum(label_strings != 'species') < 5):
            n_neighbors = 1

        if (sum(label_strings == 'species') < 3) or (sum(label_strings != 'species') < 3):
            continue

        features = data1.loc[:, ~data1.columns.isin(['Genome'])] #remove labels
        features = features.loc[:, ~features.columns.isin([
                                                        'Completeness', 
                                                        'Contamination',
                                                        'GC.content',
                                                        'Genome.size',
                                                        'domain',
                                                        'phylum',
                                                        'class',
                                                        'order',
                                                        'family',
                                                        'genus',
                                                        'species',
                                                        'taxonomic_dist',
                                                        'cultured_level'
                                                        ])]

        features = pd.get_dummies(features)
        labels = pd.get_dummies(label_strings)['species']
        #print(label_strings, labels)

        #print(features.columns)
        #print(labels)

        #print('Cultured: ', sum(labels.eq(1)), ', Uncultured: ', sum(labels.eq(0)))
        #print()

        if SHUF == 'shuffled':
            features = shuffle(features) #do random shuffle

        for i in range(10002, 10007):
            print('Pre-preprocessing data...')
            features = helpers.clean_data(features)
            X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels, test_size=0.2, random_state=i)
            X_train, y_train = helpers.perform_SMOTE(X_train, y_train, k_neighbors=n_neighbors)

            print('Running LASSO...')
            #X_train_reduced, X_test_reduced, LASSO_stats, 
            model = run_LASSO(X_train, X_test, y_train, y_test, phylum)
            #LASSO_stats.sort_values('permutation_importance', ascending=False, ignore_index=True, inplace=True)
            y_pred = model.predict(X_test)
            y_pred_binary = [0 if elem <= 0.5 else 1 for elem in y_pred]

            auc = roc_auc_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred_binary)
            prec = precision_score(y_test, y_pred_binary)
            rec = recall_score(y_test, y_pred_binary)

            f.write(phylum+':\n')
            f.write('AUC: '+str(round(auc, 3))+'\n')
            f.write('Accuracy: '+str(round(acc, 3))+'\n')
            f.write('Precision: '+str(round(prec, 3))+'\n')
            f.write('Recall: '+str(round(rec, 3))+'\n')
            f.write('\n\n')

        #full_df = full_df.append(LASSO_stats)

    #full_df.to_csv(curr_dir+'/files/cutoff_0.95/HG/HG-{}-bootstrapped-by-phylum-{}-LASSO-stats-SMOTE.csv'.format(SHUF, DATA))

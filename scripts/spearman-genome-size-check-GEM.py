import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import statistics

from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.utils import shuffle
import scipy.stats as st

import helpers

#--------------------------------------------------------------------------------------------------#

def run_LASSO(X_train_scaled, X_test_scaled, y_train, y_test, phylum=None, param_grid=None, random_state=872510):
    lasso = LogisticRegression(penalty='l1', solver='liblinear')
    model = BaggingRegressor(base_estimator=lasso, n_estimators=50, bootstrap=True, verbose=0, random_state=random_state)

    model.fit(X_train_scaled, y_train)

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

    l95 = []
    u95 = []
    sig = []
    for data in coefs.values():
         conf_it = st.t.interval(alpha=0.95, df=len(data)-1, loc=statistics.mean(data), scale=st.sem(data))

         if conf_it[0] <= 0 and conf_it[1] >= 0:
             sig.append(False)
         elif np.isnan(conf_it[0]) or np.isnan(conf_it[1]):
             sig.append(False)
         else:
             sig.append(True)

         l95.append(conf_it[0])
         u95.append(conf_it[1])

    #imp = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=random_state)

    output = pd.DataFrame()
    output['phylum_name'] = [phylum]*len(feature_names)
    output['feature_name'] = feature_names
    output['coef'] = means.values()
    output['coef_sd'] = std_devs.values()
    output['lower_95'] = l95
    output['upper_95'] = u95
    output['count'] = counts.values()
    output['significant'] = sig
    #output['permutation_importance'] = imp.importances_mean
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


#--------------------------------------------------------------------------------------------------#

DATA = 'GEM'
TYPE = 'annotation'

if __name__ == '__main__':

    print('Loading data...')
    curr_dir = os.getcwd()

    meta_file = curr_dir+'/data/GEM_data/GEM_metadata.tsv'
    path_file = curr_dir+'/data/GEM_data/pathway_features_counts_wide.tsv'
    annot_file = curr_dir+'/data/GEM_data/annotation_features_counts_wide.tsv'

    metadata = pd.read_csv(meta_file, sep='\t', header=0, encoding=helpers.detect_encoding(meta_file))

    if TYPE == 'annotation':
        annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=helpers.detect_encoding(annot_file))
        annot_features = helpers.normalize_abundances(annot_features)
        data = pd.merge(metadata, annot_features, on='genome_id', how='inner')
    else:
        path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=helpers.detect_encoding(path_file))
        path_features = helpers.normalize_abundances(path_features)
        data = pd.merge(metadata, path_features, on='genome_id', how='inner')

    phylum_list = set(list(data['phylum']))

    #full_df = pd.DataFrame(columns=['phylum_name', 'feature_name', 'coef', 'coef_sd', 'lower_95', 'upper_95', 'count', 'significant', 'permutation_importance'])
    for phylum in phylum_list:
        if pd.isna(phylum):
            continue

        data1 = data[data['phylum'] == phylum]

        if data1.shape[0] < 100:
            continue

        label_strings = data1['cultured.status']

        if len(set(list(label_strings))) != 2:
            continue

        #need at least 10 cultured labels
        if (sum(label_strings == 'cultured') < 10) or (sum(label_strings == 'uncultured') < 10):
            continue


        print(phylum, ':', data1.shape)

        top_10 = data1['completeness'].quantile(0.90)
        bottom_10 = data1['completeness'].quantile(0.10)

        top_idx = data1[data1.completeness >= top_10].index
        bottom_idx = data1[data1.completeness <= bottom_10].index

        #print(len(top_idx), len(bottom_idx))
        
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

        features = pd.get_dummies(features)
        labels = pd.get_dummies(label_strings)['cultured']
        #print(label_strings, labels)

        #features = shuffle(features) #do random shuffle

        print('Pre-preprocessing data...')
        features = helpers.clean_data(features)
        X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels, test_size=0.2)
        unique, counts = np.unique(np.array(y_test), return_counts=True)    
        #print(dict(zip(unique, counts)))

        X_train, y_train = helpers.perform_SMOTE(X_train, y_train)

        print('Running LASSO...')
        X_train_reduced, X_test_reduced, LASSO_stats, model = run_LASSO(X_train, X_test, y_train, y_test, phylum)
        #LASSO_stats.sort_values('permutation_importance', ascending=False, ignore_index=True, inplace=True)

        y_pred = model.predict(X_train)
        y_pred_binary = [0 if elem <= 0.5 else 1 for elem in y_pred]

        auc = roc_auc_score(y_train, y_pred)
        acc = accuracy_score(y_train, y_pred_binary)
        prec = precision_score(y_train, y_pred_binary)
        rec = recall_score(y_train, y_pred_binary)

        print(phylum+':\n')
        print('AUC: '+str(round(auc, 3))+'\n')
        print('Accuracy: '+str(round(acc, 3))+'\n')
        print('Precision: '+str(round(prec, 3))+'\n')
        print('Recall: '+str(round(rec, 3))+'\n')
        print('\n\n')
 

        y_pred = model.predict(X_test)
        y_pred_binary = [0 if elem <= 0.5 else 1 for elem in y_pred]

        auc = roc_auc_score(y_test, y_pred)
        acc = balanced_accuracy_score(y_test, y_pred_binary)
        prec = precision_score(y_test, y_pred_binary)
        rec = recall_score(y_test, y_pred_binary)

        print(phylum+':\n')
        print('AUC: '+str(round(auc, 3))+'\n')
        print('Accuracy: '+str(round(acc, 3))+'\n')
        print('Precision: '+str(round(prec, 3))+'\n')
        print('Recall: '+str(round(rec, 3))+'\n')
        print('\n\n')


        top_preds = model.predict(features.loc[top_idx])
        #unique, counts = np.unique(np.array(labels.loc[top_idx]), return_counts=True)    
        #print(dict(zip(unique, counts)))
        bottom_preds = model.predict(features.loc[bottom_idx])
        #unique, counts = np.unique(np.array(labels.loc[bottom_idx]), return_counts=True)    
        #print(dict(zip(unique, counts)))

        top_preds = 1 - top_preds
        #print(set(top_preds))
        bottom_preds = 1 - bottom_preds
        #print(set(bottom_preds))

        top_GS = data1.loc[top_idx, 'genome_length']
        bottom_GS = data1.loc[bottom_idx, 'genome_length']

        #print(top_preds, bottom_preds)
        #print(top_GS, bottom_GS)

        preds = np.concatenate([top_preds, bottom_preds])
        GS = np.concatenate([top_GS, bottom_GS])
        rho, p = st.spearmanr(preds, GS, nan_policy='omit')
        #rho1, p1 = st.spearmanr(top_preds, top_GS, nan_policy='omit')
        #rho2, p2 = st.spearmanr(bottom_preds, bottom_GS, nan_policy='omit')

        with open(curr_dir+'/files/{}/{}-spearman-genome-size-{}.txt'.format(DATA, DATA, TYPE), 'a') as file:
            file.write('Phylum name: '+phylum+'\n')
            file.write('Spearman Correlation of Predicted (Uncultured) Probability and Genome Length for Top/Bottom 10% Completeness - \n')
            file.write('Rho: '+str(round(rho, 3))+', P-value: '+str(round(p, 3))+'\n\n')
            #file.write('Spearman Correlation of Predicted (Uncultured) Probability and Genome Length for Bottom 10% Completeness - \n')
            #file.write('Rho: '+str(round(rho2, 3))+', P-value: '+str(round(p2, 3))+'\n\n')
            file.write('\n\n')

        #full_df = full_df.append(LASSO_stats)

    #full_df.to_csv(curr_dir+'/files/shuffled-bootstrapped-by-phylum-annotation-LASSO-stats.csv')

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import statistics

from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import scipy.stats as st

import helpers

#--------------------------------------------------------------------------------------------------#

def run_RF(X_train, X_test, y_train, y_test, image_name=None, image_path=None, param_grid=None, label=None, title=None, color=None):

    if param_grid == None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy'],
        }

    if label == None:
        label = y_train.name

    clf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=5,
        n_jobs=5,
        verbose=1
    )

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) #get probabilities for AUC
    probs = y_prob[:,1]

    print('Calculating metrics for:', label)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    print('Calculating feature importance...')
    importances = pd.DataFrame(clf.best_estimator_.feature_importances_, index=X_train.columns, columns=['importance'])

    return importances

def run_LASSO(X_train_scaled, X_test_scaled, y_train, y_test, phylum=None, param_grid=None, random_state=872510):
    lasso = LogisticRegression(penalty='l1', solver='liblinear')
    model = BaggingRegressor(base_estimator=lasso, n_estimators=100, bootstrap=True, verbose=0, random_state=random_state)

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

    imp = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=random_state)

    output = pd.DataFrame()
    output['phylum_name'] = [phylum]*len(feature_names)
    output['feature_name'] = feature_names
    output['coef'] = means.values()
    output['coef_sd'] = std_devs.values()
    output['lower_95'] = l95
    output['upper_95'] = u95
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

    return LASSO_train, LASSO_test, output


#--------------------------------------------------------------------------------------------------#


if __name__ == '__main__':

    print('Loading data...')
    curr_dir = os.getcwd()

    meta_file = curr_dir+'/data/GEM_data/GEM_metadata.tsv'
    path_file = curr_dir+'/data/GEM_data/pathway_features_counts_wide.tsv'
    annot_file = curr_dir+'/data/GEM_data/annotation_features_counts_wide.tsv'

    metadata = pd.read_csv(meta_file, sep='\t', header=0, encoding=helpers.detect_encoding(meta_file))

    #path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=helpers.detect_encoding(path_file))
    #path_features = helpers.normalize_abundances(path_features)
    annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=helpers.detect_encoding(annot_file))
    annot_features = helpers.normalize_abundances(annot_features)
    data = pd.merge(metadata, annot_features, on='genome_id', how='inner')

    phylum_list = set(list(data['phylum']))

    full_df = pd.DataFrame(columns=['phylum_name', 'feature_name', 'coef', 'coef_sd', 'lower_95', 'upper_95', 'count', 'significant', 'permutation_importance'])
    for phylum in phylum_list:
        if pd.isna(phylum):
            continue

        data1 = data[data['phylum'] == phylum]

        if data1.shape[0] < 100:
            continue

        label_strings = data1['cultured.status']

        if len(set(list(label_strings))) != 2:
            continue

        print(phylum, ':', data1.shape)

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

        print('Pre-preprocessing data...')
        features = helpers.clean_data(features)
        X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels, test_size=0.2)

        print('Running LASSO...')
        X_train_reduced, X_test_reduced, LASSO_stats = run_LASSO(X_train, X_test, y_train, y_test, phylum)
        LASSO_stats.sort_values('permutation_importance', ascending=False, ignore_index=True, inplace=True)

        full_df = full_df.append(LASSO_stats)

    full_df.to_csv(curr_dir+'/files/updated-bootstrapped-by-phylum-annotation-LASSO-stats.csv')

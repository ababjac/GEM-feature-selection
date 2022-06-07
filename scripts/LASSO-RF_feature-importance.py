import pandas as pd
import numpy as np
import os
import shap

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from chardet.universaldetector import UniversalDetector
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

#--------------------------------------------------------------------------------------------------#

def detect_encoding(file):
    detector = UniversalDetector()
    detector.reset()
    with open(file, 'rb') as f:
        for row in f:
            detector.feed(row)
            if detector.done: break

    detector.close()
    return detector.result['encoding']

#--------------------------------------------------------------------------------------------------#

def standard_scale(train, test):
    xtrain_scaled = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(StandardScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

#--------------------------------------------------------------------------------------------------#

def normalize_abundances(df): #this is for GEM only
    norm_df = pd.DataFrame()

    for c in df.columns:
        if not c.__contains__('genome_id'):
            total = df.loc[:, c].sum()

            if total == 0: #skip because there is no point in predicting these sites
                continue

            norm_df[c] = df[c] / total

    norm_df['genome_id'] = df['genome_id']
    return norm_df

#--------------------------------------------------------------------------------------------------#

def clean_data(data):
    remove = [col for col in data.columns if data[col].isna().sum() != 0]
    return data.loc[:, ~data.columns.isin(remove)] #this gets rid of remaining NA

#--------------------------------------------------------------------------------------------------#

def split_and_scale_data(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=5)
    X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

#--------------------------------------------------------------------------------------------------#

def perform_SMOTE(X, y, k_neighbors=2, random_state=1982):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm

#--------------------------------------------------------------------------------------------------#

def preload_GEM(include_metadata=True):
    curr_dir = os.getcwd()

    meta_file = curr_dir+'/data/GEM_data/GEM_metadata.tsv'
    annot_file = curr_dir+'/data/GEM_data/annotation_features_counts_wide.tsv'
    path_file = curr_dir+'/data/GEM_data/pathway_features_counts_wide.tsv'

    metadata = pd.read_csv(meta_file, sep='\t', header=0, encoding=detect_encoding(meta_file))
    annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=detect_encoding(annot_file))
    annot_features = normalize_abundances(annot_features)
    path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=detect_encoding(path_file))
    path_features = normalize_abundances(path_features)

    data = pd.merge(metadata, path_features, on='genome_id', how='inner')
    data = pd.merge(data, annot_features, on='genome_id', how='inner')

    label_strings = data['cultured.status']

    features = data.loc[:, ~data.columns.isin(['genome_id', 'cultured.status'])] #remove labels
    if not include_metadata: #remove metadata
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

    return features, labels

#--------------------------------------------------------------------------------------------------#

def run_LASSO(X_train_scaled, X_test_scaled, y_train, param_grid = None):
    if param_grid == None:
        param_grid = {'alpha':np.arange(0.01, 2, 0.05)}

    search = GridSearchCV(estimator = Lasso(),
                          param_grid = param_grid,
                          cv = 5,
                          scoring="neg_mean_squared_error",
                          verbose=3
                          )

    search.fit(X_train_scaled, y_train)
    coefficients = search.best_estimator_.coef_
    importance = np.abs(coefficients)
    remove = np.array(X_train_scaled.columns)[importance == 0]

    print('Features being removed:')
    print(remove)

    if len(remove) != len(X_train_scaled.columns):
        LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
        LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]
    else:
        LASSO_train = X_train_scaled
        LASSO_test = X_test_scaled

    return LASSO_train, LASSO_test

#--------------------------------------------------------------------------------------------------#

def plot_confusion_matrix(y_pred, y_actual, title, path, color):
    if color == None:
        color = 'Oranges'

    plt.gca().set_aspect('equal')
    cf_matrix = confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    ax = sns.heatmap(cf_matrix, annot=True, cmap=color)

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

def plot_auc(y_pred, y_actual, title, path):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

def run_RF(X_train, X_test, y_train, y_test, image_name, image_path=None, param_grid=None, label=None, title=None, color=None):

    if param_grid == None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
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
        verbose=3
    )

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Features selected by LASSO:')
    print(X_train.columns)

    print('Predicting on test data for label:', label)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) #get probabilities for AUC
    probs = y_prob[:,1]

    print('Calculating metrics for:', label)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    if image_path != None:
        if image_path[-1] != '/':
            filename = image_path+'/'+image_name
        else:
            filename = image_path+image_name
    else:
        filename = image_name #save in current directory

    if title == None:
        title = label

    print('Calculating AUC score...')
    plot_auc(y_pred=probs, y_actual=y_test, title='AUC for '+title, path=filename+'_AUC.png')

    print('Plotting:', label)
    plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=title, path=filename+'_CM.png', color=color)

    print()

    print('Calculating feature importance...')
    explainer = shap.TreeExplainer(clf.best_estimator_)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(filename+'_FI-barplot.png', bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, X_test, plot_type='dot')
    plt.savefig(filename+'_FI-summary.png', bbox_inches='tight')
    plt.close()


#--------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    print('Loading GEM...')
    features, labels = preload_GEM(include_metadata=False)

    print('Pre-preprocessing data...')
    features = clean_data(features)
    X_train, X_test, y_train, y_test = split_and_scale_data(features, labels)
    X_test, y_test = perform_SMOTE(X_test, y_test)

    print('Running LASSO...')
    X_train_reduced, X_test_reduced = run_LASSO(X_train, X_test, y_train)

    print('Running Random Forests...')
    run_RF(X_train_reduced, X_test_reduced, y_train, y_test, 'LASSO-RF-GEM-both', image_path='./figures', color='cool')

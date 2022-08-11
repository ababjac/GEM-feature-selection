import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from chardet.universaldetector import UniversalDetector
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso

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

def split_and_scale_data(features, labels, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=5)
    X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

#--------------------------------------------------------------------------------------------------#

def perform_SMOTE(X, y, k_neighbors=2, random_state=1982):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm

#--------------------------------------------------------------------------------------------------#

def write_list_to_file(filename, l):
    file = open(filename, 'w')

    for elem in l:
        file.write(elem)
        file.write('\n')

    file.close()

#--------------------------------------------------------------------------------------------------#

def preload_GEM(include_metadata=True, features_type='both', test_mode=False):

    if test_mode:
        features_type = 'pathway'

    curr_dir = os.getcwd()

    meta_file = curr_dir+'/data/GEM_data/GEM_metadata.tsv'
    annot_file = curr_dir+'/data/GEM_data/annotation_features_counts_wide.tsv'
    path_file = curr_dir+'/data/GEM_data/pathway_features_counts_wide.tsv'

    metadata = pd.read_csv(meta_file, sep='\t', header=0, encoding=detect_encoding(meta_file))

    if features_type == 'both':
        annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=detect_encoding(annot_file))
        annot_features = normalize_abundances(annot_features)
        path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=detect_encoding(path_file))
        path_features = normalize_abundances(path_features)
        data = pd.merge(metadata, path_features, on='genome_id', how='inner')
        data = pd.merge(data, annot_features, on='genome_id', how='inner')
    elif features_type == 'pathway':
        path_features = pd.read_csv(path_file, sep='\t', header=0, encoding=detect_encoding(path_file))
        path_features = normalize_abundances(path_features)
        data = pd.merge(metadata, path_features, on='genome_id', how='inner')
    elif features_type == 'annotation':
        annot_features = pd.read_csv(annot_file, sep='\t', header=0, encoding=detect_encoding(annot_file))
        annot_features = normalize_abundances(annot_features)
        data = pd.merge(metadata, annot_features, on='genome_id', how='inner')

    if test_mode:
        data = data.sample(2000)

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
    #print(labels)

    return features, labels

#--------------------------------------------------------------------------------------------------#

def preload_MALARIA(include_metadata=True):
    curr_dir = os.getcwd()

    meta1_file = curr_dir+'/data/malaria_data/mok_meta.tsv'
    meta2_file = curr_dir+'/data/malaria_data/zhu_meta.csv'
    expr_file = curr_dir+'/data/malaria_data/new_expression.csv'

    metadata1 = pd.read_csv(meta1_file, sep='\t', header=0, encoding=detect_encoding(meta1_file))
    metadata2 = pd.read_csv(meta2_file, header=0, encoding=detect_encoding(meta2_file))
    metadata1['SampleID'] = metadata1['SampleID'].str.replace('-', '.')
    metadata = pd.merge(metadata1, metadata2, on='SampleID', how='inner')

    expr_features = pd.read_csv(expr_file, header=0, index_col=0, encoding=detect_encoding(expr_file)).fillna(0)
    expr_features = expr_features.T
    expr_features = expr_features.reset_index()
    expr_features.rename(columns={'index':'GenotypeID'}, inplace=True)

    data = pd.merge(metadata, expr_features, on='GenotypeID', how='inner')

    data = data[(data['Clearance'] >= 6) | (data['Clearance'] < 5)] #these are likely "semi-resistant" samples so remove
    data['Resistant'] =  np.where(data['Clearance'] >= 6.0, 1, 0)

    features = data.loc[:, ~data.columns.isin(['Clearance', 'Resistant', 'SampleID', 'GenotypeID', 'SampleID.Pf3k', 'Parasites clearance time', 'Field_site'])] #remove labels
    if not include_metadata: #remove metadata
        features = features.loc[:, ~features.columns.isin(['FieldsiteName',
                                                           'Country',
                                                           'Hemoglobin(g/dL)',
                                                           'Hematocrit(%)',
                                                           'parasitemia',
                                                           'Parasite count',
                                                           'Sample collection time(24hr)',
                                                           'Patient temperature',
                                                           'Drug',
                                                           'ACT_partnerdrug',
                                                           'Duration of lag phase',
                                                           'PC50',
                                                           'PC90',
                                                           'Estimated HPI',
                                                           'Estimated gametocytes proportion',
                                                           'ArtRFounders',
                                                           'Timepoint',
                                                           'RNA',
                                                           'Asexual_stage',
                                                           'Lifestage',
                                                           'Long_class'
                                                           ])]

    features = pd.get_dummies(features)
    labels = data['Resistant']

    return features, labels

#--------------------------------------------------------------------------------------------------#

def run_LASSO(X_train_scaled, X_test_scaled, y_train, param_grid = None):
    if param_grid == None:
        param_grid = {'alpha':np.arange(0.01, 3, 0.05)}

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

    if len(remove) < len(X_train_scaled.columns):
        LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
        LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]
    else:
        print('LASSO COULDNT DECIDE')
        LASSO_train = X_train_scaled
        LASSO_test = X_test_scaled

    # coef = [c for c in coefficients if c != 0]
    # l = [colname+' : '+str(coefficient) for colname, coefficient in zip(list(LASSO_train.columns), coef)]
    # write_list_to_file('files/LASSO-coefficients-annotation-list.txt', l)

    return LASSO_train, LASSO_test

#--------------------------------------------------------------------------------------------------#

def plot_confusion_matrix(y_pred, y_actual, title, path, color=None):
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

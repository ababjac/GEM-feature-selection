import pandas as pd
import numpy as np
import os, sys
#import shap

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
from sklearn.inspection import PartialDependenceDisplay

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

def preload_GEM(include_metadata=True, features_type='both'):
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

def plot_feature_importance(columns, importances, path):
    plt.figure(figsize=(16,8))
    sorted_idx = importances.argsort()
    sorted_idx = [i for i in sorted_idx if importances[i] > 0.01]
    plt.barh(columns[sorted_idx], importances[sorted_idx])
    plt.xlabel('Gini Values')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

#taken from here: https://stats.stackexchange.com/questions/288736/random-forest-positive-negative-feature-importance
def calculate_pseudo_coefficients(X, y, thr, probs, importances, nfeatures, path):
    dec = list(map(lambda x: (x> thr)*1, probs))
    val_c = X.copy()

    #scale features for visualization
    val_c = pd.DataFrame(StandardScaler().fit_transform(val_c), columns=X.columns)

    val_c = val_c[importances.sort_values('importance', ascending=False).index[0:nfeatures]]
    val_c['t']=y
    val_c['p']=dec
    val_c['err']=np.NAN
    #print(val_c)

    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'] = 2#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'] = 1#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'] = 4#'fn'

    n_fp = len(val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'])
    n_tn = len(val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'])
    n_tp = len(val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'])
    n_fn = len(val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'])

    fp = np.round(val_c[(val_c['t']==0)&(val_c['p']==1)].mean(),2)
    tn = np.round(val_c[(val_c['t']==0)&(val_c['p']==0)].mean(),2)
    tp =  np.round(val_c[(val_c['t']==1)&(val_c['p']==1)].mean(),2)
    fn =  np.round(val_c[(val_c['t']==1)&(val_c['p']==0)].mean(),2)


    c = pd.concat([tp,fp,tn,fn],names=['tp','fp','tn','fn'],axis=1)
    pd.set_option('display.max_colwidth',900)
    c = c[0:-3]

    c.columns = ['TP','FP','TN','FN']

    c.plot.bar()
    plt.title('Relative Scaled Model Coefficients for True/False Positive Rates')
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

def run_RF(X_train, X_test, y_train, y_test, image_name, image_path=None, param_grid=None, label=None, title=None, color=None):

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
        verbose=3
    )

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Features selected by LASSO:')

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

    # print('Calculating feature importance...')
    # importances = pd.DataFrame(clf.best_estimator_.feature_importances_, index=X_train.columns, columns=['importance'])
    # importances = importances[importances['importance'] > 0.01]
    # plot_feature_importance(X_train.columns, clf.best_estimator_.feature_importances_, filename+'_FI-gini.png')
    # calculate_pseudo_coefficients(X_test, y_test, 0.5, probs, importances, len(X_train.columns), filename+'_FI-rates.png')

    sorted_idx = clf.best_estimator_.feature_importances_.argsort()
    PartialDependenceDisplay.from_estimator(clf.best_estimator_, X_test, [sorted_idx[0]], kind='both', target=1, centered=True)
    name = X_train.columns[sorted_idx[0]].replace(' ', '')
    plt.savefig(filename+'_PDP-'+name+'.png')
    plt.close()


#--------------------------------------------------------------------------------------------------#

def write_list_to_file(filename, l):
    file = open(filename, 'w')

    for elem in l:
        file.write(elem)
        file.write('\n')

    file.close()

#--------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 '+str(sys.argv[0])+' [pathway, annotation, both]')
        exit

    features_type_ = sys.argv[1]

    print('Loading GEM...')
    features, labels = preload_GEM(include_metadata=False, features_type=features_type_)

    print('Pre-preprocessing data...')
    features = clean_data(features)
    X_train, X_test, y_train, y_test = split_and_scale_data(features, labels)
    X_test, y_test = perform_SMOTE(X_test, y_test)

    print('Running LASSO...')
    X_train_reduced, X_test_reduced = run_LASSO(X_train, X_test, y_train)
    write_list_to_file('files/LASSO-features-'+str(features_type_)+'-list.txt', list(X_train_reduced.columns))

    print('Running Random Forests...')
    run_RF(X_train_reduced, X_test_reduced, y_train, y_test, 'LASSO-RF-GEM-'+str(features_type_), image_path='./figures', color='Blues')

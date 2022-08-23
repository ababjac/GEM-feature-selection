import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

import helpers

#--------------------------------------------------------------------------------------------------#

def run_XGBoost(X_train, X_test, y_train, y_test, image_name, image_path=None, label=None, title=None, color=None):
    if label == None:
        label = y_train.name

    clf = xgb.XGBRegressor(objective='binary:logistic', seed=5000)
    #print(X_train.shape, y_train.shape)

    print('Building model for label:', label)
    clf.fit(X_train, y_train)

    print('Predicting on test data for label:', label)
    probs = clf.predict(X_test)
    y_pred = np.where(probs >= 0.5, 1, 0)

    print('Calculating metrics for:', label)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    helpers.write_rates_csv(y_test, y_pred)
    if image_path != None:
        if image_path[-1] != '/':
            filename = image_path+'/'+image_name
        else:
            filename = image_path+image_name
    else:
        filename = image_name #save in current directory

    if title == None:
        title = label
    #
    # #print(clf.feature_importances_)
    # print('Calculating feature importance...')
    # importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
    # importances = importances[importances['importance'] > 0.01]
    # helpers.plot_feature_importance(X_train.columns, clf.feature_importances_, filename+'_FI-gini.png')
    # helpers.calculate_pseudo_coefficients(X_test, y_test, 0.5, probs, importances, len(X_train.columns), filename+'_FI-rates.png')
    #
    print('Calculating AUC score...')
    helpers.plot_auc(y_pred=probs, y_actual=y_test, title='AUC for '+title, path=filename+'_AUC.png')

    print('Plotting:', label)
    helpers.plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=title, path=filename+'_CM.png', color=color)


#--------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 '+str(sys.argv[0])+' [pathway, annotation, both]')
        exit()

    features_type_ = sys.argv[1]

    print('Loading GEM...')
    features, labels = helpers.preload_GEM(include_metadata=False, features_type=features_type_)
    #features, labels = helpers.preload_MALARIA()#include_metadata=False)

    print('Pre-preprocessing data...')
    features = helpers.clean_data(features)
    X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels)
    X_test, y_test = helpers.perform_SMOTE(X_test, y_test)

    print('Running XGBoost...')
    run_XGBoost(X_train, X_test, y_train, y_test, 'XGBoost-GEM-'+str(features_type_), image_path='./figures/XGBoost', color='Blues')

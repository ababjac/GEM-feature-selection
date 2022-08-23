import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay

import helpers

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
    helpers.plot_auc(y_pred=probs, y_actual=y_test, title='AUC for '+title, path=filename+'_AUC.png')

    print('Plotting:', label)
    helpers.plot_confusion_matrix(y_pred=y_pred, y_actual=y_test, title=title, path=filename+'_CM.png', color=color)

    print()
    helpers.write_rates_csv(y_test, y_pred)

    # print('Calculating feature importance...')
    # importances = pd.DataFrame(clf.best_estimator_.feature_importances_, index=X_train.columns, columns=['importance'])
    # importances = importances[importances['importance'] > 0.01]
    # helpers.plot_feature_importance(X_train.columns, clf.best_estimator_.feature_importances_, filename+'_FI-gini.png')
    # helpers.calculate_pseudo_coefficients(X_test, y_test, 0.5, probs, importances, len(X_train.columns), filename+'_FI-rates.png')

    # sorted_idx = clf.best_estimator_.feature_importances_.argsort()[::-1]
    # #print(sorted_idx)
    # for i in range(10):
    #     PartialDependenceDisplay.from_estimator(clf.best_estimator_, X_test, [sorted_idx[i]], target=0, centered=True)
    #     name = X_train.columns[sorted_idx[i]].replace(' ', '').replace('/', '_')
    #     #print(name)
    #     plt.savefig(filename+'_PDP-'+name+'.png')
    #     plt.close()


#--------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    # if len(sys.argv) < 3:
    #     print('Usage: python3 '+str(sys.argv[0])+' [pathway, annotation, both]')
    #     exit
    #
    features_type_ = sys.argv[1]

    print('Loading data...')
    features, labels = helpers.preload_GEM(include_metadata=False, features_type=features_type_)
    #features, labels = helpers.preload_MALARIA(include_metadata=False)

    print('Pre-preprocessing data...')
    features = helpers.clean_data(features)
    X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels)
    X_test, y_test = helpers.perform_SMOTE(X_test, y_test)

    print('Running LASSO...')
    X_train_reduced, X_test_reduced = helpers.run_LASSO(X_train, X_test, y_train)
    #helpers.write_list_to_file('files/LASSO-features-malaria-list.txt', list(X_train_reduced.columns))

    print('Running Random Forests...')
    run_RF(X_train_reduced, X_test_reduced, y_train, y_test, 'LASSO-RF-GEM-'+str(features_type_), image_path='./figures/LASSO-RF', color='Blues')

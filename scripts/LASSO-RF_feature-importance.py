import pandas as pd
import numpy as np
import os, sys
import helpers

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 '+str(sys.argv[0])+' [pathway, annotation, both]')
        exit

    features_type_ = sys.argv[1]

    print('Loading GEM...')
    features, labels = helpers.preload_GEM(include_metadata=False, features_type=features_type_)

    print('Pre-preprocessing data...')
    features = helpers.clean_data(features)
    X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels)
    X_test, y_test = helpers.perform_SMOTE(X_test, y_test)

    print('Running LASSO...')
    X_train_reduced, X_test_reduced = helpers.run_LASSO(X_train, X_test, y_train)
    helpers.write_list_to_file('files/LASSO-features-'+str(features_type_)+'-list.txt', list(X_train_reduced.columns))

    print('Running Random Forests...')
    helpers.run_RF(X_train_reduced, X_test_reduced, y_train, y_test, 'LASSO-RF-GEM-'+str(features_type_), image_path='./figures', color='Blues')

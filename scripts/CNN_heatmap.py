import pandas as pd
import numpy as np
import os, sys
import helpers

from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score

#simple cross product to transform each row into an "image" for CNN
def transform_features(features):
    T = np.array([[1], [1], [1], [1], [1]])
    # print(T.shape)
    #
    # row = np.array([features.iloc[0]])
    # print(row.shape)

    features['dots'] = features.apply(lambda row: np.matmul(T, np.array([row])), axis=1)
    return features['dots']

#based on tutorial from here: https://opendatascience.com/building-a-custom-convolutional-neural-network-in-keras/
def build_CNN():
    my_model= models.Sequential()

    my_model.add(Conv2D(16, (1, 3), activation='relu', padding='same',
                    input_shape=(5,70,2)))
    my_model.add(MaxPooling2D(1, 2))#, padding='same')

    my_model.add(Conv2D(32, (1, 3), activation='relu', padding='same'))
    my_model.add(MaxPooling2D((1, 2)))#, padding='same'))

    my_model.add(Conv2D(64, (1, 3), activation='relu', padding='same'))
    my_model.add(MaxPooling2D((1, 2)))#, padding='same'))

    my_model.add(Conv2D(128, (1, 3), activation='relu', padding='same'))
    my_model.add(MaxPooling2D((1, 2)))#, padding='same'))

    my_model.add(GlobalAveragePooling2D())

    my_model.add(Dense(64, activation='relu'))
    my_model.add(BatchNormalization())

    my_model.add(Dense(2, activation='sigmoid'))

    return my_model


if __name__ == '__main__':
    label = 'Cultured'
    #if len(sys.argv) < 2:
    #    print('Usage: python3 '+str(sys.argv[0])+' [pathway, annotation, both]')
    #    exit

    #features_type_ = sys.argv[1]

    print('Loading GEM...')
    features, labels = helpers.preload_GEM(include_metadata=False, test_mode=True)

    print('Pre-preprocessing data...')
    features = helpers.clean_data(features)

    X_train, X_test, y_train, y_test = helpers.split_and_scale_data(features, labels)
    X_train, X_val, y_train, y_val = helpers.split_and_scale_data(X_train, y_train, test_size=0.1)

    X_test, y_test = helpers.perform_SMOTE(X_test, y_test) #rebalance class labels for prediction task

    X_train_new = transform_features(X_train)
    X_val_new = transform_features(X_val)
    X_test_new = transform_features(X_test)

    model = build_CNN()
    es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc= ModelCheckpoint('models/test.h5', monitor='val_loss',
                    mode='min', verbose=1, save_best_only=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_new, y_train, validation_data=(X_val_new, y_val))

    print('Predicting on test data for label:', label)
    y_pred = model.predict(X_test_new)
    y_prob = model.predict_proba(X_test_new) #get probabilities for AUC
    probs = y_prob[:,1]

    print('Calculating metrics for:', label)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    filename = 'figures/CAM/CAM-test'

    print('Calculating AUC score...')
    helpers.plot_auc(y_pred=probs, y_actual=y_test, title='AUC for '+label, path=filename+'_AUC.png')

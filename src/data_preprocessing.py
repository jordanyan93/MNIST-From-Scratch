# src/data_preprocessing.py
import pandas as pd
import numpy as np

def load_data():
    numbers_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    return numbers_df, test_df

def preprocess_data(numbers_df, test_df, sample_size=10000):
    data = np.array(numbers_df)
    test = np.array(test_df)
    
    data_train = data[:sample_size].T
    data_dev = data[sample_size:sample_size + 5000].T

    Y_train = one_hot_array(data_train[0])
    X_train = data_train[1:]
    Y_dev = one_hot_array(data_dev[0])
    X_dev = data_dev[1:]

    X_test = test.T

    return X_train, Y_train, X_dev, Y_dev, X_test

def one_hot_array(Y):
    b = np.zeros((Y.size, Y.max() + 1))
    b[np.arange(Y.size), Y] = 1
    return b.T

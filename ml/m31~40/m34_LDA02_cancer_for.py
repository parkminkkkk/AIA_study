#RFC디폴트 값 : RFC_LDA값 비교 

import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_diabetes, fetch_covtype
from sklearn.metrics import accuracy_score

# 1. 데이터

data_list = [load_iris, load_breast_cancer, load_digits, fetch_covtype]
data_name = ['iris', 'cancer', 'digits', 'fetch']

# Create empty lists for results
results_no_lda = []
results_with_lda = []


for i, v in enumerate(data_list):
    x,y =v(return_X_y=True)
    print("="*10, data_name[i] ,"="*10)

    # Version without LDA
    x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=640, train_size=0.8, shuffle=True)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    model_no_lda = RandomForestClassifier(random_state=640)
    model_no_lda.fit(x_train, y_train)
    result_no_lda = model_no_lda.score(x_test, y_test)
    print('result (no LDA):', result_no_lda)
    results_no_lda.append(result_no_lda)
    
    # Version with LDA
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=640, train_size=0.8, shuffle=True)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model_with_lda = RandomForestClassifier(random_state=640)
    model_with_lda.fit(x_train, y_train)
    result_with_lda = model_with_lda.score(x_test, y_test)
    print('result (with LDA):', result_with_lda)
    results_with_lda.append(result_with_lda)
print("==============================================")
print("Results without LDA:", results_no_lda)
print("Results with LDA:", results_with_lda)


'''
========== iris ==========
result (no LDA): 0.9666666666666667
result (with LDA): 0.9666666666666667
========== cancer ==========
result (no LDA): 0.9649122807017544
result (with LDA): 0.9649122807017544
========== digits ==========
result (no LDA): 0.9638888888888889
result (with LDA): 0.9638888888888889
========== fetch ==========
result (no LDA): 0.9572214142492018
result (with LDA): 0.9572214142492018
==============================================
'''
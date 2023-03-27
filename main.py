
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from engine import *
from classifiers import *
from performance import *

df = pd.read_csv("../input/autism_screening.csv", na_values=['?'])

X, y = split_to_X_y(df)

# From the Exploratory Data Analysis
print()
print("The outlier on the Age column/feature, where Age is ", X.at[52,'age'] )
X.at[52,'age'] = X.age.median()
print("The outlier on the Age column/feature is now assigned to the MEDIAN of the Age =  ", X.at[52,'age'] )
print()

y_transform = y_cat_transform_num(y)

X_train_set_stratified, X_test_set_stratified, y_train_set_stratified, y_test_set_stratified = stratify_split(X, y_transform)

X_train_fit_transfm, X_test_transfm = preprocess(X_train_set_stratified, X_test_set_stratified)

# To see if a model overfits or underfits the Test set
for name, clf in zip(label, classifier):
    clf.fit( X_train_fit_transfm, y_train_set_stratified )
    
    print( "Training Accuracy: {0: .6f} [ {1} is used. ]".format( 
        clf.score( X_train_fit_transfm, y_train_set_stratified ), 
        name ) )

    print( "Test Accuracy: {0: .6f} [ {1} is used. ] \n ".format( 
        clf.score( X_test_transfm, y_test_set_stratified ), 
        name ) )

fold = 10

cross_value_score_trainSet_testSet(label,
                                   classifier,
                                   X_train_fit_transfm,
                                   y_train_set_stratified,
                                   X_test_transfm,
                                   y_test_set_stratified,
                                   fold)


f1_score_trainSet_testSet(label,
                          classifier,
                          X_train_fit_transfm,
                          y_train_set_stratified,
                          X_test_transfm,
                          y_test_set_stratified)

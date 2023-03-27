

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Classifiers
LogReg = LogisticRegression(random_state = 42, solver = 'lbfgs')
KNNClf = KNeighborsClassifier()
DecTreeClf = DecisionTreeClassifier(random_state = 42)
AdaBoostClf = AdaBoostClassifier(random_state = 42)

classifier = [ LogReg , KNNClf, DecTreeClf, AdaBoostClf ]

label = ['Logistic Regression',
         'K Neighbors Classifier',
         'Decision Tree Classifier',
         'AdaBoost Classifier']


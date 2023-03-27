
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def split_to_X_y(DataFrame):
    y_column_target = input('Type the y or target column name (Answer is Class/ASD): ')
    y = DataFrame[y_column_target]
    X_drop_column = input('Type the feature or column in X to be dropped (Answer is result): ')
    X = DataFrame.drop(columns = [y_column_target, X_drop_column] )

    return X, y

def y_cat_transform_num(y):
    y_label_encoder = LabelEncoder()
    y_label_encoder.fit(y)
    y_transform = y_label_encoder.transform(y)

    return y_transform

def stratify_split(X, y):
    strat_X_train_set, strat_X_test_set, strat_y_train_set, strat_y_test_set = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
        )
    return strat_X_train_set, strat_X_test_set, strat_y_train_set, strat_y_test_set


def preprocess(X_training, X_test):

    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    cat_pipeline = Pipeline([
            ("ordinal_encoder", OrdinalEncoder( handle_unknown = "use_encoded_value", unknown_value = -1 ) ),    
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder( handle_unknown = "ignore") )
        ])

    num_attribs = X_training.select_dtypes(include=[np.number]).columns

    cat_attribs = X_training.select_dtypes(include=[object]).columns

    preprocess = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])

    preprocess.fit(X_training)

    X_training_fit_transform = preprocess.transform(X_training)

    X_test_transform = preprocess.transform(X_test)

    return X_training_fit_transform, X_test_transform

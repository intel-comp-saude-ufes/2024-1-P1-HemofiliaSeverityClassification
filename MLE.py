# Other imports
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

# Standard Python libraries
import time
import psutil
import numpy as np
from scipy import stats

# Third-party Library Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    KFold,
    cross_validate,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# XGBoost and LightGBM imports
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class MLE:
    def __init__(self, X, y, random_state=42):
        self.X = X
        self.y = y
        self.X_len = self.X.shape[1]
        self.y_len = len(np.unique(self.y))
        
        self.random_state = random_state
        self.outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=random_state)
        self.inner_cv = StratifiedKFold(n_splits=4, shuffle=False)
        
        self.scoring = ["accuracy", "roc_auc_ovr", "f1_macro", "recall_macro"]
        self.refit = "accuracy"
        
        # Get number of physical CPUs
        self.n_cores = psutil.cpu_count(logical=False) // 2

        self.classifiers = {
            'Dummy': DummyClassifier(random_state=self.random_state),
            'GaussianNB': GaussianNB(),
            'KNeighbors': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(random_state=self.random_state),
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'AdaBoost': AdaBoostClassifier(random_state=self.random_state),
            'Bagging': BaggingClassifier(random_state=self.random_state),
            'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state),
            'XGBoost': XGBClassifier(seed=self.random_state, nthread=1, verbosity=0),
            'LightGBM': LGBMClassifier(seed=self.random_state, nthread=1, verbosity=-1),
            'SVM': SVC(random_state=self.random_state,probability=True),
            'MLP': MLPClassifier(random_state=self.random_state)
        }
        
        self.scalers = {
            'None': None,
            'Standard': StandardScaler(),
        }

        # Select numeric columns
        self.numeric_columns = X.select_dtypes(include='number').columns

        # Get indices of numeric columns
        self.numeric_features = [X.columns.get_loc(col) for col in self.numeric_columns]

        # Identify categorical columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        # Transform categorical columns into one-hot encoded variables
        self.X = pd.get_dummies(self.X, columns=self.categorical_columns)
        
        # Convert only the one-hot encoded columns to boolean values
        one_hot_columns = self.X.columns.difference(X.columns)
        self.X[one_hot_columns] = self.X[one_hot_columns].astype(bool)


        print(self.X)
        
        if not pd.api.types.is_numeric_dtype(y):
            # Transform the target column to numeric values
            label_encoder = LabelEncoder()
            self.y = label_encoder.fit_transform(y)
            self.number_to_class = {index: label for index, label in enumerate(label_encoder.classes_)}
        print(self.y)
    
    def estimator_validate_without_param_search(self, scaler_name, estimator_name, params={}):
        print(scaler_name, estimator_name, params, flush=True)
        scaler = self.scalers[scaler_name]
        estimator = self.classifiers[estimator_name]
        
        # Define parametros
        n_jobs = self.n_cores
        verbose = 0
        
        # Transformador para as features numericas (sem aplicar nos binarios)
        numeric_transformer = Pipeline(steps=[('scaler', scaler)])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_features)])
        
        # Define pipeline com os pré-processamentos necessários
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])
        # pipeline = Pipeline(steps=[('scaler', scaler), ('estimator', estimator)])
        
        pipeline.set_params(**params)
        
        # Avalia usando ciclo EXTERNO com cross validation
        scores = cross_validate(estimator=pipeline, X=self.X, y=self.y, cv=self.outer_cv, scoring=self.scoring, n_jobs=n_jobs, verbose=verbose)
        
        acc_scores = scores['test_accuracy']
        roc_scores = scores['test_roc_auc_ovr']
        f1_scores = scores['test_f1_macro']
        recall_scores = scores['test_recall_macro']
        
        results = {
            'scaler': scaler_name,
            'classifier': estimator_name,
            'params': params,
            
            'accuracy std': acc_scores.std(),
            'accuracy': acc_scores.mean(),
            'roc_auc_ovr': roc_scores.mean(),
            'f1_macro': f1_scores.mean(),
            'recall_macro': recall_scores.mean(),
            
            'accuracy_scores': acc_scores,
        }

        return results
    
    def add_estimator_prefix(self,param_grid):
        prefixed_param_grids = {}
        temp = "estimator__"
        prefixed_param_grids = {temp + str(key): val for key, val in param_grid.items()}
        return prefixed_param_grids
    
    def estimator_validate(self, scaler_name, estimator_name, params={}):
        scaler = self.scalers[scaler_name]
        estimator = self.classifiers[estimator_name]
        
        # param definition
        n_jobs = self.n_cores
        verbose = 0
        if params == {}:
            params = self.add_estimator_prefix(param_grids[estimator_name])

        print(scaler_name, estimator_name, params, flush=True)
        start = time.time()
        
        # Numeric features scaler
        numeric_transformer = Pipeline(steps=[('scaler', scaler)])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_features)])
        
        # Pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])
        
        # Inner Cycle
        gs = GridSearchCV(estimator=pipeline, param_grid=params, cv=self.inner_cv, scoring="accuracy", n_jobs=n_jobs)
        
        # Outer Cyclo
        cv_scores = cross_validate(estimator=gs, X=self.X, y=self.y, cv=self.outer_cv, scoring=self.scoring, n_jobs=n_jobs, verbose=verbose, return_estimator=True)
        
        acc_scores = cv_scores['test_accuracy']
        roc_scores = cv_scores['test_roc_auc_ovr']
        f1_scores = cv_scores['test_f1_macro']
        recall_scores = cv_scores['test_recall_macro']
        
        results = {
            'scaler': scaler_name,
            'classifier': estimator_name,
            'params': params,
            
            'accuracy std': acc_scores.std(),
            'accuracy': acc_scores.mean(),
            'roc_auc_ovr': roc_scores.mean(),
            'f1_macro': f1_scores.mean(),
            'recall_macro': recall_scores.mean(),
            
            'accuracy_scores': acc_scores,
        }

        end = time.time()
        print(f"Time elapsed: {end - start}s")
        return results, cv_scores

param_grids = {
    'Dummy': {
        'strategy': ['stratified', 'most_frequent', 'prior', 'uniform']
    },
    'GaussianNB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1, 10]
    },
    'Bagging': {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.3, 0.7, 1.0],
        'subsample': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 70],
        'max_depth': [-1, 10, 20, 30],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}
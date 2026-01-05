# package imports 
from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json
from datetime import datetime
# for explicit typecasting 
from typing import Dict, List, Tuple, Optional, Any

# writing in OOP for good practice

# ==========================================================
class DataProcessor:

    def __init__(self, filepath:str) -> None:
        self.filepath:str = filepath
        self.label_encoders = {} # dict to store output of the encoding loop
        self.feature_names = None

# ----------------------------------------------------------
    def data_load_preprocess(self):
        df = pd.read_csv(self.filepath)
        X = df.drop(['chrun', 'customer_id'], axis=1)
        y: pd.Series = df['churn']

        # hard coded list of categorical columns in the dataset
        cat_cols = [
            'contract_type',
            'payment_method',
            'internet_service',
            'tech_support'
        ]

        # loop over cat cols and create the encodings 
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            # save the output of the encoder to the instance variable
            self.label_encoders[col] = le

        # save feature names to the instance var
        self.feature_names = list(X.columns)
        
        # final return 
        return X, y
        

# ==========================================================
class ModelTrainer:
    """Trains and evaluates a machine learning model for churn prediction.
    """

    def __init__(self, model_type:str='RandomForestClassifier', **kwargs) -> None:
        self.model_type = model_type
        self.model = self._init_model(**kwargs) #init the model and store it as an instance var

# ----------------------------------------------------------
    def _init_model(self, **kwargs) -> Any:
        """Initializes the model based on the specified type. Currently supports only RandomForestClassifier.

        Raises:
            ValueError: if anything else than "randomForestClassifier is passed.

        Returns:
            RandomForestClassifer: Created model with args passed as **kwargs.
        """

        if self.model_type == 'RandomForestClassifier':
            return RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f'Unsupported model type:\n{self.model_type}')
    
# ----------------------------------------------------------
    def train(self, X_train:pd.DataFrame, y_train:pd.Series) -> Any:
        """Trains the model on the provided training data.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target vector.

        Returns:
            object: Trained model instance.
        """

        self.model.fit(X_train, y_train)
        return self.model

# ----------------------------------------------------------
    def eval(self, X_test:pd.DataFrame, y_test:pd.Series) -> Dict[str, Any]:
        """_summary_

        Args:
            X_test (pd.DataFrame): Test freature matrix.
            y_test (pd.Series): Test target vector.

        Returns:
            Dict[str, Any]: Dict of evaluation metrics (accuracy, roc_auc) and a classification report.
        """

        # predict
        y_pred: np.ndarray = self.model.predict(X_test)
        # display class probabilities prediciton (needed for roc)
        y_pred_proba: np.ndarray = self.model.predict_proba(X_test)[:,1]  
        
        # create a dict with metrics
        metrics: Dict[str, Any] = {
            'accuracy': accuracy_score(y_test, y_pred),
            # area under reciever operating characteristic score
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(
                y_test,
                y_pred,
                output_dict=True
            )
        }
        return metrics

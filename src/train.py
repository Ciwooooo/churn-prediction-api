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
class DatapreProcessor:

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

# ==========================================================
class ModelPipeline:
    def __inint__(self, filepath, model_type='RandomForestClassifier', **model_kwargs):
        self.preprocessor = DataPreprocessor(filepath)
        self.trainer = ModelTrainer(model_type, **model_kwargs)

# ----------------------------------------------------------
    def main(self):
        print("Starting model training...")

        # Preprocess and train-test split
        X, y = self.preprocessor.data_load_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
            )
        print(f'Training set size: {len(X_train)}')
        print(f'Test set size: {len(X_test)}')

        # train and eval
        model = self.trainer.train(X_train, y_train)
        metrics = self.trainer.evaluate(X_test, y_test)

        print(f'\nModel Performance:')
        print(f'Accuracy: {metrics['accuracy']:.4f}')
        print(f'ROC-AUC:: {metrics['roc_auc']:.4f}')

        # save martifacts (model and metrics)
        self._save_artifacts(model, metrics)
    
# ----------------------------------------------------------
    def _save_artifacts(self, model, metrics):
        """Internal use func that saves the model, feature names and It's metadata.

        Args:
            model (Object): trained model
            metrics (Dict[str, any]): metrics generated by ModelTrainer.eval()
        Returns:
            None
        """
        joblib.dump(model, 'models/churn_model.pkl')
        joblib.dump(self.preprocessor.label_encoders, 'models/feature_names.pkl')

        # create the metadata dict
        metadata = {
            'training_date': datetime.now.isoformat(),
            'model_type': self.trainer.model_type,
            'n_features': len(self.preprocessor.feature_names),
            'feature_names': self.preprocessor.feature_names,
            metrics: {
                'accuracy': float(metrics['accuracy']),
                'roc_auc': float(metrics['roc_auc'])
            }
        }
        # dump into the json file
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nModel saved successfully!")
        print("Files created:")
        print("  - models/churn_model.pkl")
        print("  - models/label_encoders.pkl")
        print("  - models/feature_names.pkl")
        print("  - models/model_metadata.json")


# ==========================================================

if __name__ == '__main__':
    pipeline = ModelPipeline(
        filepath='data/churn_data.csv',
        model_type='RandomForestClassifier',
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1 # max number of jobs possible
    )
pipeline.main()
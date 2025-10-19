# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:14:30 2025

@author: julia
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score
import joblib

import sys

class RFR_class: 

    def __init__(self, threshold = 150): 
        self.random_state = 42
        self.df = None # dataframe
        self.X = None # Feature columns of original dataset
        self.y= None # Target column of original dataset

        self.X_tr = None # scaled training data
        self.y_tr = None # target column for training data
        self.X_tst = None # scaled testing data
        self.y_tst = None # target column for testing data
        self.threshold = threshold # treshold value for high-risk flag
        self.model = None


    def predict(self, split=0.3):
        """Run all other methods and return calculated mse
        of Linear Regression model

        Returns:
            float: Mean Squared Error
        """
        self._load_data()
        self._train_test_split(split)
        
        # Fit model
        model = RandomForestRegressor(max_depth= 5, 
                                      max_features = 'sqrt',
                                      min_samples_leaf = 2, 
                                      n_estimators = 200,
                                      random_state = self.random_state)       
        model.fit(self.X_tr, self.y_tr)


        # Predict testing data + calculate evaluation scores
        prediction = model.predict(self.X_tst)
        rmse = root_mean_squared_error(self.y_tst, prediction)
        
        high_risk_flag_true = (self.y_tst >= self.threshold).astype(int)
        high_risk_flag_pred = (prediction >= self.threshold).astype(int)

        precision = precision_score(high_risk_flag_true, high_risk_flag_pred)
        recall = recall_score(high_risk_flag_true, high_risk_flag_pred)

        joblib.dump(model, "RFR_regression_model.joblib")
         
        # Log metrics
        self._log_metrics(rmse, precision, recall)

        return rmse, precision, recall


    def _load_data(self):
        """Obtains dataset from sklearn diabetes.
        Separates target column into Y from the other columns into X
        """

        self.df = load_diabetes(as_frame=True)
        self.X = self.df.frame.drop(columns=["target"])
        self.y = self.df.frame["target"]
        

    def _train_test_split(self, test_size = 0.3):
        """Make a train test split of the data and return the scaled with Standard scaler

        Args:
            test_size (float, optional): Percentage used for test data. Defaults to 0.3.
        """

        self.X_tr, self.X_tst, self.y_tr, self.y_tst  = train_test_split(self.X,
                                                            self.y,
                                                            test_size=test_size,
                                                            random_state=self.random_state)   
    def _log_metrics(self, rmse, precision, recall):
        with open("CHANGELOG.md", "a") as f:
            f.write("\n### Regression v0.2 + High-risk Flag\n")
            f.write(f"- RMSE: {rmse:.4f}\n")
            f.write(f"- Precision (high-risk): {precision:.4f}\n")
            f.write(f"- Recall (high-risk): {recall:.4f}\n")
            
            
if __name__ == "__main__":
    RFR = RFR_class()
    
    if len(sys.argv) < 2:
        print("[!] Error no input value provided using default 0.3 split")
        print(f"[+] RMSE, precision, recall: {RFR.predict()}")
        sys.exit(1)

    input_value = float(sys.argv[1])
    print(f"[+] RMSE, precision, recall: {RFR.predict(input_value)}")
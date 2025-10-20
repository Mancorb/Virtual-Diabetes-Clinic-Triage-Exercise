#%% Imports
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score
import joblib
import uvicorn
import pandas as pd

#%% RandomForestRegressor class
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
        self.msre = None
        self.precision = None
        self.recall = None

        self.feature_order = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]


    def predict(self, data):
        # Convert dict to DataFrame with proper column names
        X = pd.DataFrame([data], columns=self.feature_order)
        self._train_model()  # or ensure model is trained

        self._calculate_metrics()

        prediction = self.model.predict(X)
        return float(prediction[0])


    def _train_model(self, split=0.3):
        """Train the model with the provided data and traintest split

        Args:
            split (float, optional): percentage for train test split. Defaults to 0.3.
        """
        self._load_data()
        self._train_test_split(split)

        # Fit model
        self.model = RandomForestRegressor(max_depth= 5, 
                                      max_features = 'sqrt',
                                      min_samples_leaf = 2, 
                                      n_estimators = 200,
                                      random_state = self.random_state)       
        self.model.fit(self.X_tr, self.y_tr)

        joblib.dump(self.model, "RFR_regression_model.joblib")


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
    
    
    def _calculate_metrics(self):
        prediction = self.model.predict(self.X_tst)
        self.rmse = root_mean_squared_error(self.y_tst, prediction)

        high_risk_flag_true = (self.y_tst >= self.threshold).astype(int)
        high_risk_flag_pred = (prediction >= self.threshold).astype(int)

        self.precision = precision_score(high_risk_flag_true, high_risk_flag_pred)
        self.recall = recall_score(high_risk_flag_true, high_risk_flag_pred)

        self._log_metrics()


    def _log_metrics(self):
        with open("CHANGELOG.md", "a") as f:
            f.write("\n### Regression v0.2 + High-risk Flag\n")
            f.write(f"- RMSE: {self.rmse:.4f}\n")
            f.write(f"- Precision (high-risk): {self.precision:.4f}\n")
            f.write(f"- Recall (high-risk): {self.recall:.4f}\n")
            
            
if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0",
        port=8000,
        reload=True
    )
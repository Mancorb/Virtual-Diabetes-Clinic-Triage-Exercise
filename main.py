from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import joblib
import sys

class LR_class: 

    def __init__(self):
        pass
        self.random_state = 42
        self.X_tr = None # scaled training data
        self.y_tr = None # target column for training data
        self.X_tst = None # scaled testing data
        self.y_tst = None # target column for testing data


    def predict(self, split=0.3):
        """Run all other methods and return calculated mse
        of Linear Regression model

        Returns:
            float: Mean Squared Error
        """
        self._load_data()
        self._train_test_split(split)

        model = LinearRegression()

        #(Trining data, training data labels)
        model.fit(self.X_tr,self.y_tr)

        #Predict testing data
        prediction = model.predict(self.X_tst)
        
        joblib.dump(model, "LR_regression_model.joblib")
         
        # Log metrics
        rmse = root_mean_squared_error(self.y_tst, prediction)
        self._log_metrics(rmse)
        return rmse


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

        X_train, X_test, self.y_tr, self.y_tst  = train_test_split(self.X,
                                                            self.y,
                                                            test_size=test_size,
                                                            random_state=self.random_state)
        scaler = StandardScaler()
        self.X_tr = scaler.fit_transform(X_train)
        self.X_tst = scaler.transform(X_test)
  
    def _log_metrics(self, rmse):
        with open("CHANGELOG.md", "a") as f:
            f.write("\n### Regression v0.1\n")
            f.write(f"- RMSE: {rmse:.4f}\n")
        
if __name__ == "__main__":
    LR = LR_class()
    
    if len(sys.argv) < 2:
        print("[!] Error no input value provided using default 0.3 split")
        print(f"[+] RMSE: {LR.predict()}")
        sys.exit(1)

    input_value = float(sys.argv[1])
    print(f"[+] RMSE: {LR.predict(input_value)}")
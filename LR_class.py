from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
import uvicorn


# ----- MODEL CLASS -----
class LR_class: 

    def __init__(self):
        self.random_state = 42
        self.df = None # dataframe
        self.X = None # Feature columns of original dataset
        self.y= None # Target column of original dataset

        self.X_tr = None # scaled training data
        self.y_tr = None # target column for training data
        self.X_tst = None # scaled testing data
        self.y_tst = None # target column for testing data
        self.model = None
        self.rmse = None
        self.feature_order = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
   

    def _train_model(self, split=0.3):
        """Train the Linear Regression model with specified split

        Args:
            split (float, optional): Train test split percentage. Defaults to 0.3.
        """
        self._load_data()
        self._train_test_split(split)

        self.model = LinearRegression()

        #(Trining data, training data labels)
        self.model.fit(self.X_tr,self.y_tr)


    def _calculate_metrics(self):
        """Calculate RMSE and save in local variable"""
        #Predict testing data
        prediction = self.model.predict(self.X_tst)

        self.rmse = root_mean_squared_error(self.y_tst, prediction)


    def _log_metrics(self):
        """
        Save information on JSon fileon stats of the model

        Args:
            rmse (float): rmse of the currently running model
        """
        self._calculate_metrics()

        with open("CHANGELOG.md", "w") as f:
            f.write("\n### Root Mean Squared Metric v0.1\n")
            f.write(f"- RMSE: {self.rmse:.4f}\n")
            


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


    def predict(self, data):
        self._train_model()
        self._log_metrics
        X_list = [data[feat] for feat in self.feature_order]
        X = np.array(X_list).reshape(1, -1)

        return float(self.model.predict(X)[0])
    
if __name__ =="__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0",
        port=8000,
        reload=True
    )
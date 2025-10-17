from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import sys

class Random_Forest_class: 

    def __init__(self):
        pass


if __name__ == "__main__":
    #LR = LR_class()
    
    if len(sys.argv) < 2:
        print("[!] Error no input value provided using default 0.3 split")
        #print(f"[+] RMSE: {LR.predict()}")
        sys.exit(1)

    input_value = float(sys.argv[1])
    #print(f"[+] RMSE: {LR.predict(input_value)}")
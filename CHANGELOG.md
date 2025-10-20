### Metrics
### Regression v0.1
- RMSE: 53.1202

### Regression v0.2 + High-risk Flag
- RMSE: 52.3314
- Precision (high-risk): 0.7708
- Recall (high-risk): 0.74


### Explanation
### Changes v0.2
- **Model**: Replaced 'LinearRegression' with 'RandomForestRegressor' to better capture nonlinear relationships in the data
- **Hyperparameters**: Tuned and decided using 'GridSearchCV' to find the optimal balance between model accuracy and generalization
- **Preprocessing**: Removed 'StandardScalar' as tree models handle unscaled features, and the dataset is already standardized
- **High-Risk flag**: Added a threshold-based flagging system for high-risk patients
- **Metrics**: Logged additional evaluation metrics for the flagging classification; 'Precision' and 'Recall'

### Conclusions
- **RMSE**: RMSE improved slightly (~1.5%) from v0.1 to v0.2, showing small but improved regression accuracy
- **Precision and Recall**: Precision and Recall metrics demonstrate a useful model for identifying high-risk patients
-**API Update**: In version v0.2, the API now returns both a continuous risk score and a flag indicating the presence of high risk, thereby improving clinical interpretability
- **Reproducibility**: All models were trained and tested with fixed random seeds (random_state = 42)
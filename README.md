
### Developed By: Thanjiyappan k
### Register No: 212222240108

# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
### Date: 

### AIM:
To Compute the AutoCorrelation Function (ACF) of the power Consumption dataset and 
to determine the model
type to fit the data.
### ALGORITHM:
1.Import the necessary packages.
2.Calculate the mean and variance of the data.
3.Implement normalization by scaling the data to have a mean of 0 and a variance of 1.
4.Compute the correlation and store the results in an array.
5.Represent the result graphically.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set seed for reproducibility
np.random.seed(0)

# Load and preprocess data
data = pd.read_csv('ev.csv')
data['year'] = pd.to_datetime(data['year'])
data = data.sort_values(by='year')
data.set_index('year', inplace=True)
data.dropna(inplace=True)

# Plot the consumption data
plt.figure(figsize=(12, 6))
plt.plot(data['value'], label='Data')
plt.xlabel('year')
plt.ylabel('value')
plt.legend()
plt.title('')
plt.show()

# Split into train and test data
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
y_train = train_data['value']
y_test = test_data['value']

# Compute and plot ACF for the first 35 lags
plt.figure(figsize=(12, 6))
plot_acf(data['value'], lags=35)
plt.title('ACF of Consumption Data (First 35 Lags)')
plt.show()
# Fit an autoregressive model (AR)
lag_order = 1  # you can adjust based on the ACF plot
data['value'].corr(data['value'].shift(1))
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
lag_order = 35 
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()

# Predictions
y_pred = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Compute MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
variance = np.var(y_test)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Variance_testing: {variance:.2f}')
```

### OUTPUT:
#### VISUAL REPRESENTATION OF DATASET:
![image](https://github.com/user-attachments/assets/a4af76fb-ba47-415e-b899-369a33c61a75)


#### AUTO CORRELATION:
![image](https://github.com/user-attachments/assets/7b820e12-e2b8-476a-8aed-4a9b7642426d)

#### VALUES OF MAE,RMSE,VARIANCE:
![image](https://github.com/user-attachments/assets/dc42464a-1ffa-4961-96e2-c2a1c54a32eb)



### RESULT: 
Thus, The python code for implementing auto correlation for power consumption is successfully executed.

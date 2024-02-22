# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRASANTH U
RegisterNumber:  212222220031
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV data
df = pd.read_csv('/content/Book1.csv')

# View the beginning and end of the data
df.head()
df.tail()

# Segregate data into variables
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
# Create a linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values using the model
y_pred = regressor.predict(x_test)

# Display predicted and actual values
print("Predicted values:", y_pred)
print("Actual values:", y_test)

# Visualize the training data
plt.scatter(x_train, y_train, color="black")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title("Hours VS scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Visualize the test data
plt.scatter(x_test, y_test, color="cyan")
plt.plot(x_test, regressor.predict(x_test), color="green")
plt.title("Hours VS scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print('RMSE = ', rmse)
```

## Output:
## df.head():
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/02200271-b89f-47ce-85c0-7ac333a3f5ad)

## df.tail():
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/b01fcfe6-331c-4bd2-a748-5171f35aceee)

## Array value of X:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/33c8c759-0f52-4fa4-bef3-0508a042d0bd)

## Array value of y:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/fe1f06b3-4a42-48b0-813c-801d7f64b986)

## Values of Y prediction:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/647218a4-4caa-46ba-acad-7b69cf8f7f0a)

## Values of Y test:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/2fb54720-0a5d-447b-b05e-81fd95b7d120)

## Training Set Graph:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/383114ff-2102-45b4-9ad9-7c59fbc0bf04)

## Test Set Graph:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/1a3e47d4-c3c1-4c2c-a79b-84088f199d37)

## Values of MSE, MAE and RMSE:
![image](https://github.com/Prasanth9025/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343686/34d3493a-9476-47e9-bddf-96098a22ace6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

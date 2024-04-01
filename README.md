# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries and load the dataset.
2. Data Preprocessing and split the Data into Training and Testing Sets.
3. Train the Decision Tree Regressor Model and make Predictions.
4. Evaluate the Model.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Gokkul M
RegisterNumber: 212223240039 
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
data = pd.read_csv("salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())
x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
dt = DecisionTreeRegressor()
print(dt.fit(x_train, y_train))
y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print(mse)
r2 = metrics.r2_score(y_test, y_pred)
print(r2)
print(dt.predict([[5, 6]]))
```
## Output:
![image](https://github.com/Gokkul-M/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870543/45467293-1418-4e43-af99-e2e74009e910)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

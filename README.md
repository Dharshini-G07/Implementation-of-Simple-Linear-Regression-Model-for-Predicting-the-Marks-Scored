# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Priyadharshini .G
RegisterNumber: 212224230209  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: Priyadharshini G")
print("Reg no : 212224230209")
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: Priyadharshini G")
print("Reg no : 212224230209")
```

## Output:

## head()
![image](https://github.com/user-attachments/assets/169ba70b-a69f-46dd-9c94-b45711dff4d2)
## tail()
![image](https://github.com/user-attachments/assets/407d0490-d456-41ad-a251-b581719afd0b)
## Array
![image](https://github.com/user-attachments/assets/e81a3b1e-9ecf-480d-b533-f28c5ca9e724)
![image](https://github.com/user-attachments/assets/c13a7273-2633-4f32-8c50-ef590514d305)
## Errors
![image](https://github.com/user-attachments/assets/17a29cc1-841d-4883-bb17-3c14cc5b03a3)
## Hours vs Scores training set
![image](https://github.com/user-attachments/assets/27230549-5ee0-48c0-a2c0-1e61b764832a)
## Hours vs Scores test set
![image](https://github.com/user-attachments/assets/bb81cbea-81ca-46dd-b87e-c92869a7d827)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

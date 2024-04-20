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
6. Compare the graphs and hence we obtained the linear regression for the
given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: kailash s m  
RegisterNumber: 212222040068 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1.csv')
df.head()
df.tail()
#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying predicted values
y_pred
#displaying actual values
y_test
#graph plot for training data
plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
df.head():
 ![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/34a5d955-db1e-4c15-9cf7-c33dffc4cdb8)

df.tail():
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/3217d8f6-e32f-4f04-b312-a9921df58d31)


Array value of X:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/f2cc20ae-3374-453c-804d-671621aca509)

Array value of Y:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/a9e79f64-fd91-4dc0-bdd1-a387a04598e7)

Values of Y prediction:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/a7cbb95d-8a74-47fd-8745-1e2d1b1ac035)

Values of Y test:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/229fcef4-b7f4-4aed-9dca-280fc5e2fea4)

Training Set Graph:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/5544f494-cda6-4b98-96ed-e64b8c4d2169)


Test Set Graph:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/59a9a47e-c8c3-4f20-9642-00b715284f48)

Values of MSE, MAE and RMSE:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123893976/5a248a40-719f-4bb2-b7cc-c89234353321)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

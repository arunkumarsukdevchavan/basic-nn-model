# EX-01 Developing a Neural Network Regression Model
### Aim:
To develop a neural network regression model for the given dataset.&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE: 19.08.2024**

### Theory:
Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error.

### Neural Network Model:
![image](https://github.com/user-attachments/assets/151f56b9-8129-4253-a9c3-744ab9c77732)

### Design Steps:

- STEP 1:Loading the dataset
- STEP 2:Split the dataset into training and testing
- STEP 3:Create MinMaxScalar objects ,fit the model and transform the data.
- STEP 4:Build the Neural Network Model and compile the model.
- STEP 5:Train the model with the training data.
- STEP 6:Plot the performance plot
- STEP 7:Evaluate the model with the testing data.

## Program:
#### Developed By: ARUN KUMAR SUKDEV CHAVAN - 212222230013

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df=pd.read_csv("data.csv")
df.head()
df.describe()
df.info()
X=df[["Input"]].values
Y=df[["Output"]].values
xtrain,xtest,ytrain,ytest=tts(X,Y,test_size=0.3,random_state=0)
scaler=MinMaxScaler()
scaler.fit(xtrain)
xtrainscaled=scaler.transform(xtrain)
model=Sequential([Dense(units=4,activation='relu',input_shape=[1]),
                  Dense(units=6,activation='relu'),
                  Dense(units=4,activation='relu'),
                  Dense(units=1)])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrainscaled,ytrain,epochs=2000)
loss=pd.DataFrame(model.history.history)
loss.plot()
xtestscaled=scaler.transform(xtest)
model.evaluate(xtestscaled,ytest)
p=[[5]]
pscale= scaler.transform(p)
model.predict(pscale)

```
## Output:

### Dataset Information:
##### df.head()
![image](https://github.com/user-attachments/assets/329cb717-0233-4c38-bd6e-4e9177f5f75c)




##### df.info()
![image](https://github.com/user-attachments/assets/3232e518-82ec-4c70-8bae-267360534722)




##### df.describe()
![image](https://github.com/user-attachments/assets/770fb456-9eb2-4b5b-894f-2cfb8befb6df)






##### Training Loss Vs Iteration Plot:
![image](https://github.com/user-attachments/assets/7afac966-cd56-4227-89cb-3fd4e4f4b8a4)




##### Test Data Root Mean Squared Error:
![image](https://github.com/user-attachments/assets/2dd87cd4-5e97-4f6e-91dc-33a1aaa6d2d7)





##### New Sample Data Prediction:
![image](https://github.com/user-attachments/assets/58bdd753-ac31-418c-99e0-213ce5e46417)





### Result:
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.



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
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Activation Function').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'int'})
dataset1 = dataset1.astype({'Output':'int'})
dataset1.head()
dataset1.describe()
dataset1.info()
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
model=Sequential([Dense(units=4,activation='relu',input_shape=[1]),
                  Dense(units=6,activation='relu'),
                  Dense(units=4,activation='relu'),
                  Dense(units=1)])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=2000)
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
![image](https://github.com/user-attachments/assets/cc4d377a-5799-467f-97b5-3caad6b4616f)



##### df.info()
![image](https://github.com/user-attachments/assets/01c1b9b9-874e-493f-8582-0c1003903679)



##### df.describe()
![image](https://github.com/user-attachments/assets/f2f79d2d-b67e-4d58-bd09-6b3227142060)





##### Training Loss Vs Iteration Plot:
![image](https://github.com/user-attachments/assets/a00140d4-1748-4823-a30e-8c8befb6eaec)



##### Test Data Root Mean Squared Error:
![image](https://github.com/user-attachments/assets/875f5cf9-5440-4a6a-8183-dfb1d994fb6d)




##### New Sample Data Prediction:
![image](https://github.com/user-attachments/assets/83c615aa-8eab-4b9f-93fa-bedb1d2791b5)




### Result:
Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.



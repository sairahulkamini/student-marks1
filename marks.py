import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv('marks2.csv')
X = df.iloc[:,:-1].values  
y = df.iloc[:, 1].values
train_data, test_data, train_target, test_target = train_test_split(df[["hours","hours_st","iq","loc"]], df['marks'], test_size=0.3,random_state=1)
model = LinearRegression()
model.fit(train_data, train_target)
hourstudy=float(8)
hourslept=float(7)
iq=float(105)
loc=float(1)
marks=model.predict([[hourstudy,hourslept,iq,loc]])
print("Marks:",marks)
accuracy = model.score(test_data, test_target)
print('Accuracy:', accuracy)

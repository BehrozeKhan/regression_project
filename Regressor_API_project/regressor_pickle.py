import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle

df = pd.read_csv(r'Regression_project/abalone_original.csv')
print(df)
#Apply one hot encoder on Sex column
sex = pd.get_dummies(df.sex,prefix = 'Sex')
print(sex)
#Join it with the original dataset
df = df.join(sex)
print("===============New Data================")
print(df)
#Delete the previous column
df.drop(['sex'],axis=1,inplace=True)
print(df)
f = df.columns.get_loc
x = df.iloc[:, np.r_[f('length'),f('diameter'),f('height'),f('shucked-weight'),f('viscera-weight'),f('shell-weight'),f('rings'),f('Sex_F'),f('Sex_I'),f('Sex_M')]]
y = df.iloc[:,np.r_[f('whole-weight')]]
print("===============X-Data==================")
print(x)
print("===============Y-Data==================")
print(y)
#Split the data into Trainng and Testing datasets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.7,random_state=0)
print("===================X_Train Data==================")
print(x_train)
print("====================X_Test Data======================")
print(x_test)
print("=================Y_Train Data=================")
print(y_train)
print("==========================y_Test Data==========================")
print(y_test)
print("==============Decision Tree Regressor==================")
dt = DecisionTreeRegressor()
dtreg = dt.fit(x_train,y_train)
pickle.dump(dtreg,open('regressor.pkl', 'wb'))
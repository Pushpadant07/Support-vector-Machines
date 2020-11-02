import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn import preprocessing

SalaryTest = pd.read_csv("D:\\ExcelR Data\\Assignments\\Support vector Machines\\SalaryData_Test.csv")
SalaryTrain = pd.read_csv("D:\\ExcelR Data\\Assignments\\Support vector Machines\\SalaryData_Train.csv")
#test data
SalaryTest.head()
SalaryTest.describe()
SalaryTest.columns
#train data
SalaryTrain.head()
SalaryTrain.describe()
SalaryTrain.columns

###################### Creating Dummy Variables using "Lable Encoder" ###############################
Le = preprocessing.LabelEncoder()
# SalaryTest
SalaryTest['Workclass']=Le.fit_transform(SalaryTest['workclass'])
SalaryTest['Education'] = Le.fit_transform(SalaryTest['education'])
SalaryTest['Educationno'] = Le.fit_transform(SalaryTest['educationno'])
SalaryTest['Maritalstatus'] = Le.fit_transform(SalaryTest['maritalstatus'])
SalaryTest['Occupation'] = Le.fit_transform(SalaryTest['occupation'])
SalaryTest['Relationship'] = Le.fit_transform(SalaryTest['relationship'])
SalaryTest['Race'] = Le.fit_transform(SalaryTest['race'])
SalaryTest['Sex'] = Le.fit_transform(SalaryTest['sex'])
SalaryTest['Native'] = Le.fit_transform(SalaryTest['native'])



#SalaryTrain
SalaryTrain['Workclass']=Le.fit_transform(SalaryTrain['workclass'])
SalaryTrain['Education'] = Le.fit_transform(SalaryTrain['education'])
SalaryTrain['Educationno'] = Le.fit_transform(SalaryTrain['educationno'])
SalaryTrain['Maritalstatus'] = Le.fit_transform(SalaryTrain['maritalstatus'])
SalaryTrain['Occupation'] = Le.fit_transform(SalaryTrain['occupation'])
SalaryTrain['Relationship'] = Le.fit_transform(SalaryTrain['relationship'])
SalaryTrain['Race'] = Le.fit_transform(SalaryTrain['race'])
SalaryTrain['Sex'] = Le.fit_transform(SalaryTrain['sex'])
SalaryTrain['Native'] = Le.fit_transform(SalaryTrain['native'])

########### Droping the unwanted columns ###########################
# SalaryTest
SalaryTest.drop(["workclass"],inplace=True,axis=1)
SalaryTest.drop(["education"],inplace=True,axis=1)
SalaryTest.drop(["educationno"],inplace=True,axis=1)
SalaryTest.drop(["maritalstatus"],inplace=True,axis=1)
SalaryTest.drop(["occupation"],inplace=True,axis=1)
SalaryTest.drop(["relationship"],inplace=True,axis=1)
SalaryTest.drop(["race"],inplace=True,axis=1)
SalaryTest.drop(["sex"],inplace=True,axis=1)
SalaryTest.drop(["native"],inplace=True,axis=1)

#SalaryTrain
SalaryTrain.drop(["workclass"],inplace=True,axis=1)
SalaryTrain.drop(["education"],inplace=True,axis=1)
SalaryTrain.drop(["educationno"],inplace=True,axis=1)
SalaryTrain.drop(["maritalstatus"],inplace=True,axis=1)
SalaryTrain.drop(["occupation"],inplace=True,axis=1)
SalaryTrain.drop(["relationship"],inplace=True,axis=1)
SalaryTrain.drop(["race"],inplace=True,axis=1)
SalaryTrain.drop(["sex"],inplace=True,axis=1)
SalaryTrain.drop(["native"],inplace=True,axis=1)

from sklearn.svm import SVC

train_X = SalaryTrain.iloc[:,[0,1,2,3,5,6,7,8,9,10,11,12,13]]
train_y = SalaryTrain.iloc[:,4]
test_X = SalaryTest.iloc[:,[0,1,2,3,5,6,7,8,9,10,11,12,13]]
test_y  = SalaryTest.iloc[:,4]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
# help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) 
# Accuracy = 98.076%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) 
# Accuracy = 96.794%

# kernel = rbf- (radial base funciton)
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)
# Accuracy = 75%
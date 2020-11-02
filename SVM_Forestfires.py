import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn import preprocessing

Forestfires = pd.read_csv("D:\\ExcelR Data\\Assignments\\Support vector Machines\\forestfires.csv")

Forestfires.head()
Forestfires.describe()
Forestfires.columns

Le = preprocessing.LabelEncoder()
#convertong catogorical to numerical
Forestfires['Month']=Le.fit_transform(Forestfires['month'])
Forestfires['Day']=Le.fit_transform(Forestfires['day'])
#Droping
Forestfires.drop(["month"],inplace=True,axis=1)
Forestfires.drop(["day"],inplace=True,axis=1)

colnames = list(Forestfires.columns)
pred=Forestfires.drop("size_category",axis=1)
predictors = preprocessing.normalize(pred)#input
target = colnames[28] #output

sns.boxplot(x="size_category",y="wind",data=Forestfires,palette = "hls")
sns.boxplot(x="rain",y="size_category",data=Forestfires,palette = "hls")
sns.boxplot(x="size_category",y="area",data=Forestfires,palette = "hls")
sns.boxplot(x="size_category",y="temp",data=Forestfires,palette = "hls")


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(Forestfires,test_size = 0.3)
test.head()
trn = train.iloc[:,0:]
train_X = trn.drop("size_category",axis=1)
train_y = train.iloc[:,28]
tst = test.iloc[:,0:]
test_X = tst.drop("size_category",axis=1)
test_y  = test.iloc[:,28]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
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
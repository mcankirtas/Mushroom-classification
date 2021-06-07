import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

veriler = pd.read_csv("mushrooms.csv")

veriler.drop(["veil-type"], inplace=True, axis=1)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in veriler.columns[0:] :
    veriler[i] = le.fit_transform(veriler[i])
    

x=veriler.iloc[:,1:]
y=veriler.iloc[:,0:1]

corelation_matrix = x.corr()

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)    

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.svm import SVC

svr_reg = SVC(kernel="poly")
svr_reg.fit(X_train, y_train.values.ravel())

predicted_svr=svr_reg.predict(X_test)

cm_svm = confusion_matrix(y_test, predicted_svr)
acc_svm = accuracy_score(y_test, predicted_svr)

#decision tree
from sklearn.tree import DecisionTreeClassifier

r_dt = DecisionTreeClassifier(criterion="gini", splitter= "best", max_features="auto", max_depth=4)

r_dt.fit(x_train,y_train)

r_dt_predicted = r_dt.predict(x_test)

cm_dt = confusion_matrix(y_test, r_dt_predicted)

acc_dt = accuracy_score(y_test, r_dt_predicted)

from sklearn import tree

fig = plt.figure(figsize=(25,20))
tree.plot_tree(r_dt, filled=True)

#random forest
from sklearn.ensemble import RandomForestClassifier

rf_reg = RandomForestClassifier(n_estimators=40, criterion= "entropy", max_depth=3)
                                
rf_reg.fit(X_train,y_train.values.ravel())

predicted_rf = rf_reg.predict(X_test)

cm_rf = confusion_matrix(y_test, predicted_rf)
acc_rf = accuracy_score(y_test, predicted_rf)


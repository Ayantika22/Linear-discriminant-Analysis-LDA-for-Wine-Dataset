#!/usr/bin/env python
# coding: utf-8

# # Program: Clustering for Wine dataset using LDA

# In[1]:


import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn import datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading Wine Dataset 

wine = datasets.load_wine()

X = wine.data
y = wine.target
target_names = wine.target_names


# In[8]:


# fitting the LDA model
lda = LDA(n_components=2)
lda_X = lda.fit(X,y).transform(X)


# In[9]:


plt.scatter(lda_X[y == 0, 0], lda_X[y == 0, 1], s =80, c = 'orange', label = 'Type 0')
plt.scatter(lda_X[y == 1, 0], lda_X[y == 1, 1], s =80,  c = 'yellow', label = 'Type 1')
plt.scatter(lda_X[y == 2, 0], lda_X[y == 2, 1], s =80,  c = 'green', label = 'Type 2')
plt.title('LDA plot for Wine Dataset')
plt.legend()


# In[3]:


# Importing Libraries for Modelling.
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[20]:



# Assigning values of X and y from dataset


X=wine.iloc[:,:-1].values
y=wine.iloc[:,-1].values


#Setting training and testing values

Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

# Modeling is done using KNN classifiers.
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)


# Display the Output

print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Confusion matrix \n',  confusion_matrix(y_test, y_pred))
print('Classification \n', classification_report(y_test, y_pred))


# In[21]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[6]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# # Logistic regression Accuracy

# In[7]:


#Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Logistic Regression :")
print("Accuracy = ", accuracy)
print(cm)


# # LR Cohen Kappa Accuracy

# In[8]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # K-Nearest Neighbors Accuracy

# In[9]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("K Nearest Neighbors :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for KNN

# In[10]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Support Vector Machine Accuracy

# In[47]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)


# In[48]:


#Support Vector Machine
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Support Vector Machine:")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for SVM

# In[49]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Gaussian Naive Bayes Accuracy

# In[37]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# In[38]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Gaussian Naive Bayes :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for GNB

# In[39]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Decision Tree Classifier Accuracy

# In[40]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[41]:


#Decision Tree Classifier
from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

classifier = DT(criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for DTC

# In[42]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Random Forest Classifier Accuracy

# In[43]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF
classifier = RF(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for RFC

# In[44]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[ ]:





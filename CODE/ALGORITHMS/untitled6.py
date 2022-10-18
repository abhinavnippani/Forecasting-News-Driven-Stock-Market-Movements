import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

#path = 'C:/Users/prash/Downloads/STOCK MARKET/CODE/'
#os.chdir(path + 'ALGORITHMS/')


path = 'C:/Users/prash/Downloads/STOCK MARKET/'
os.chdir(path + 'CODE/ALGORITHMS/')
#from Functions import training_algorithm


X = pd.read_csv(path + 'DATASETS/' + 'X_2019.csv')
Y = pd.read_csv(path + 'DATASETS/' + 'Y_2019.csv')
encoded_Y = pd.read_csv(path + 'DATASETS/' + 'encoded_Y_2019.csv')


X = np.array(X.iloc[:,2:])


Y = np.array(Y.iloc[:,1:])
Y = Y.reshape(Y.size,1)


encoded_Y = np.array(encoded_Y.iloc[:,1:])


def normalize_dataset(df):
    
    mean = df.mean(axis=0)
    df -= mean
    std = df.std(axis=0)
    df /= std
    
    return df


def xyz(X,Y,n_train_hours,p):
    
    from sklearn.model_selection import train_test_split
    
    X_train = X[:n_train_hours, :]
    Y_train = Y[:n_train_hours, :]
    X_test = X[n_train_hours:, :]
    Y_test = Y[n_train_hours:, :]
    
    #train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=(100-p)/100, random_state=42)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #train_X = normalize_dataset(train_X)
    #test_X = normalize_dataset(test_X)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multiclass import OneVsRestClassifier
    
    regressor = OneVsRestClassifier(XGBClassifier())
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    
    Y_pred = np.array(Y_pred)
    Y_pred = Y_pred.reshape((-1,Y_test.shape[1]))
    
    correct = sum(np.sum((Y_pred == Y_test),axis=1) == Y_test.shape[1])
    #for i in range(len(Y_test)):
    #    if((Y_pred>=0.5).astype(int)[i]==Y_test[i]):
    #        correct += 1
            
            
    acc = (correct/len(Y_test))*100
    
    return acc
    








acc_combined = []
acc_stock = []
acc_news = []

from xgboost import XGBClassifier


low = 70
high = 90

for i in range(low,high,1):
    
    n_train_hours = int(i*X.shape[0]/100)
    #n_train_hours = 122
    
    acc_stock.append(xyz(X[:,0:6],encoded_Y,n_train_hours,i))
    acc_combined.append(xyz(X,encoded_Y,n_train_hours,i))
    #acc_news.append(xyz(X[:,7:],Y,n_train_hours,i))
    
plt.plot(range(low,high,1),acc_combined,c='red',label = 'Stock + News')    
plt.plot(range(low,high,1),acc_stock,c='blue',label = 'Stock')
#plt.plot(range(low,high,1),acc_news,c='green',label = 'News')
plt.legend(loc="upper right")
plt.ylabel('Accuracy')
plt.xlabel('Percentage of Data Trained')
plt.title('ACC FOR DIFFERENT % OF DATA TRAINED')
#plt.figure(figsize=(40, 40))
#plt.savefig(path + 'OUTPUT/' + 'Grid_Search_Acc.png')



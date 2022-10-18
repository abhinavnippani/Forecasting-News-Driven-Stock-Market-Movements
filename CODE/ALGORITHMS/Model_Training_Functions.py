def normalize_dataset(df):
    
    mean = df.mean(axis=0)
    df -= mean
    std = df.std(axis=0)
    df /= std
    
    return df









def SimpleNN_Algo(X_train,Y_train,X_test):
    
    from keras.models import Sequential
    from keras.layers import Dense
    import warnings
    warnings.filterwarnings("ignore")
    
    model = Sequential()
    model.add(Dense(5, input_dim=X_train.shape[1], 
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, Y_train,
              batch_size=20, 
              epochs=10, 
              validation_split = 0.1, 
              verbose=1 )
    
    Y_pred = model.predict(X_test, verbose=1)
    
    return Y_pred


def RandomForest_Algo(X_train,Y_train,X_test):
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    warnings.filterwarnings("ignore")
    
    regressor = OneVsRestClassifier(RandomForestRegressor())
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    
    return Y_pred



def XGBoost_Algo(X_train,Y_train,X_test):
    
    ##### Import libraries ######
    
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    import warnings
    warnings.filterwarnings("ignore")
    
    ###### Model ########
    
    # Define the Model
    regressor = OneVsRestClassifier(XGBClassifier())
    # Train the model
    regressor.fit(X_train, Y_train)
    # Predict the values
    Y_pred = regressor.predict(X_test)
    
    ##########
    
    return Y_pred




def Metric_1_model(X,Y,algorithm = 'XGBoost',training_period = 20,step = 10,days_test = 1):
    
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    acc = []
    
    max_training_period = min(2300,len(X)-days_test)
    
    for j in range(training_period,max_training_period,step):
        #print('**************  ' + str(j) + '  **************')
        
        # Define Parameters
        period = j
        correct = 0
        total = 0
    
        for i in range(len(X) - period - days_test):
    
            #print(i+ period + 1)
            
            # Normalize the Datasets
            sc = StandardScaler()
            X = sc.fit_transform(X)
            
            #X = normalize_dataset(X)
            
            # Divide into training and testing dataframes
            X_train = X[i:(i + period + 1), :]
            Y_train = Y[i:(i + period + 1), :]
            X_test = X[(i+ period + 1):, :]
            Y_test = Y[(i+ period + 1):, :]
            
            
            #print(X_test)
    
            #X_train = normalize_dataset(X_train)
            #X_test = normalize_dataset(X_test)
            
            # Predict the Values according to the model
            if(algorithm == 'XGBoost'):
                Y_pred = XGBoost_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'RandomForest'):
                Y_pred = RandomForest_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'SimpleNN'):
                Y_pred = SimpleNN_Algo(X_train,Y_train,X_test)
            
            Y_pred = np.array(Y_pred)
            Y_pred = Y_pred.reshape((-1,Y_test.shape[1]))
            
             
            # Find the total number of testing cases
            total += 1
            
            if (np.sum((Y_pred == Y_test),axis=1) == Y_test.shape[1])[days_test-1] == 1:
                correct += 1
            # Find the number of correct predicted test cases
            #if(sum(((Y_pred>=0.5).astype(int)[days_test - 1]) == Y_test[days_test - 1,:]) == Y_train.shape[1]):
            #    correct += 1
    
        # Append the Accuracy
        acc.append((correct/total)*100)
        
        #print('*************************************************************')
        
    return acc
    


def Metric_2_model(X,Y,algorithm = 'XGBoost',low = 70, high = 90):

    from sklearn.preprocessing import StandardScaler
    import numpy as np    
    
    acc = []
    for i in range(low,high,1):
        
        n_train_hours = int(i*X.shape[0]/100)

        # Divide into training and testing dataframes
        X_train = X[:n_train_hours, :]
        Y_train = Y[:n_train_hours, :]
        X_test = X[n_train_hours:, :]
        Y_test = Y[n_train_hours:, :]
        
        # Normalize the Datasets
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        #train_X = normalize_dataset(train_X)
        #test_X = normalize_dataset(test_X)
        
        # Predict the Values according to the model
        if(algorithm == 'XGBoost'):
            Y_pred = XGBoost_Algo(X_train,Y_train,X_test)
        elif(algorithm == 'RandomForest'):
            Y_pred = RandomForest_Algo(X_train,Y_train,X_test)
        elif(algorithm == 'SimpleNN'):
            Y_pred = SimpleNN_Algo(X_train,Y_train,X_test)
        
        
        Y_pred = np.array(Y_pred)
        Y_pred = Y_pred.reshape((-1,Y_test.shape[1]))
        
        correct = sum(np.sum((Y_pred == Y_test),axis=1) == Y_test.shape[1]) 
        
        #correct = 0
        #for i in range(len(Y_test)):
        #    if(sum(((Y_pred>=0.5).astype(int)[0]) == Y_test[0]) == Y_train.shape[1]):
        #        correct += 1
                       
        # Append the Accuracy        
        acc.append((correct/len(Y_test))*100)

    
    return acc
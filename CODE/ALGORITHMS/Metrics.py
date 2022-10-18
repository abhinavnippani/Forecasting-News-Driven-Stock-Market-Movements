def Metric_1_model(X,Y,max_training_period,algorithm = 'XGBoost',training_period = 20,step_model = 10,days_test = 1):
    
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import os
    
    path = 'C:/Users/prash/Downloads/STOCK MARKET/'

    os.chdir(path + 'CODE/ALGORITHMS/')
    from Algorithms import XGBoost_Algo,RandomForest_Algo,KNN_Algo
    from Algorithms import DecisionTree_Algo,LinearSVM_Algo,KernelSVM_Algo  
    
    # How may Timesteps back the Input data should go
    lookback = 1
    # The Period(in Timesteps) in which the data is sampled
    step = 1
    # How many Timesteps in the Future the Targets should be 
    delay = -1
    batch_size = 128
    
    
    acc = []

    
    for j in range(training_period,max_training_period,step_model):
        print('**************  ' + str(j) + '  **************')
        
        # Define Parameters
        period = j
        correct = 0
        total = 0
    
        for i in range(0,len(X) - period - days_test,5):
    
            print(str(i+ period + 1))
            
            
            
            # Normalize the Datasets
            sc = StandardScaler()
            X = sc.fit_transform(X)
            
            # Divide into training and testing dataframes
            X_train = X[i:(i + period + 1), :]
            Y_train = Y[i:(i + period + 1), :]
            X_test = X[(i+ period + 1) + (days_test - 1), :].reshape(1,-1)
            Y_test = Y[(i+ period + 1) + (days_test - 1), :].reshape(1,-1)
                
            if(algorithm == 'LSTM'):
                
                # Normalize the Datasets
                sc = StandardScaler()
                X = sc.fit_transform(X)
                
                X_formatted,Y_formatted = dataset_format_generator(X,Y,min_index = 0,
                                           max_index = X.shape[0]-delay,
                                           batch_size = batch_size,lookback = lookback,
                                           step = step,delay = delay,is_batch = False)
                
                
                # Divide into training and testing dataframes
                X_train = X_formatted[:(i + period + 1), :, :]
                Y_train = Y_formatted[:(i + period + 1), :]
                X_test = X_formatted[(i+ period + 1) + (days_test - 1), :, :].reshape(1,X_formatted.shape[1],X_formatted.shape[2])
                Y_test = Y_formatted[(i+ period + 1) + (days_test - 1), :].reshape(1,-1)
            
            #print(X_test)
            
            # Predict the Values according to the model
            if(algorithm == 'XGBoost'):
                Y_pred = XGBoost_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'RandomForest'):
                Y_pred = RandomForest_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'DecisionTree'):
                Y_pred = DecisionTree_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'LinearSVM'):
                Y_pred = LinearSVM_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'KernelSVM'):
                Y_pred = KernelSVM_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'KNN'):
                Y_pred = KNN_Algo(X_train,Y_train,X_test)
            elif(algorithm == 'LSTM'):
                
                Y_pred = LSTM_Algo(X_train,Y_train,X_test)
            
            Y_pred = np.array(Y_pred)
            Y_pred = Y_pred.reshape((-1,Y_test.shape[1]))
            
            #print(np.sum((Y_pred == Y_test),axis=1) == Y_test.shape[1])
            #print(Y_pred)
            #print(Y_test)
             
            # Find the total number of testing cases
            total += 1
            
            if (np.sum((Y_pred == Y_test),axis=1) == Y_test.shape[1]) == 1:
                correct += 1

    
        # Append the Accuracy
        acc.append((correct/total)*100)
        
        print('Acc = ',(correct/total)*100)
        
    return acc






def Metric_2_model(X,Y,algorithm = 'XGBoost',low = 70, high = 90,lookback = 1):

    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import os
    import tensorflow as tf
    
    path = 'C:/Users/prash/Downloads/STOCK MARKET/'

    os.chdir(path + 'CODE/ALGORITHMS/')
    from Algorithms import XGBoost_Algo,RandomForest_Algo,KNN_Algo
    from Algorithms import DecisionTree_Algo,LinearSVM_Algo,KernelSVM_Algo
    from Algorithms import LSTM_Algo,LSTM_2_Algo,BiLSTM_Algo,BiLSTM_2_Algo
    from Algorithms import GRU_Algo,GRU_2_Algo,BiGRU_Algo,BiGRU_2_Algo
    from Algorithms import RNN_Algo,RNN_2_Algo,BiRNN_Algo,BiRNN_2_Algo


    # How may Timesteps back the Input data should go
    #lookback = 20
    # The Period(in Timesteps) in which the data is sampled
    step = 1
    # How many Timesteps in the Future the Targets should be 
    delay = -1
    batch_size = 128

    
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
                
            
        # Predict the Values according to the model
        if(algorithm == 'XGBoost'):
            Y_pred = XGBoost_Algo(X_train,Y_train,X_test)
            
        elif(algorithm == 'RandomForest'):
            Y_pred = RandomForest_Algo(X_train,Y_train,X_test)
            #Y_pred = Y_pred.reshape((-1,Y_test.shape[1]))
        elif(algorithm == 'DecisionTree'):
            Y_pred = DecisionTree_Algo(X_train,Y_train,X_test)
        elif(algorithm == 'LinearSVM'):
            Y_pred = LinearSVM_Algo(X_train,Y_train,X_test)
        elif(algorithm == 'KernelSVM'):
            Y_pred = KernelSVM_Algo(X_train,Y_train,X_test)
        elif(algorithm == 'KNN'):
            Y_pred = KNN_Algo(X_train,Y_train,X_test)
        else:
            print("\n" + str(i) + "\n")
            
            X_train,Y_train = dataset_format_generator(X_train,Y_train,min_index = 0,
                                           max_index = X_train.shape[0]-delay,
                                           batch_size = batch_size,lookback = lookback,
                                           step = step,delay = delay,is_batch = False)
            
            X_test,Y_test = dataset_format_generator(X_test,Y_test,min_index = 0,
                                           max_index = X_test.shape[0]-delay,
                                           batch_size = batch_size,lookback = lookback,
                                           step = step,delay = delay,is_batch = False)
            
            if(algorithm == 'LSTM'):
                #tf.keras.backend.clear_session()
                Y_pred = LSTM_Algo(X_train,Y_train,X_test)
            if(algorithm == 'LSTM_2'):
                #tf.keras.backend.clear_session()
                Y_pred = LSTM_2_Algo(X_train,Y_train,X_test)
            if(algorithm == 'BiLSTM'):
                #tf.keras.backend.clear_session()
                Y_pred = BiLSTM_Algo(X_train,Y_train,X_test)
            if(algorithm == 'BiLSTM_2'):
                #tf.keras.backend.clear_session()
                Y_pred = BiLSTM_2_Algo(X_train,Y_train,X_test)
                
            if(algorithm == 'GRU'):
                Y_pred = GRU_Algo(X_train,Y_train,X_test)
            if(algorithm == 'GRU_2'):
                Y_pred = GRU_2_Algo(X_train,Y_train,X_test)
            if(algorithm == 'BiGRU'):
                Y_pred = BiGRU_Algo(X_train,Y_train,X_test)
            if(algorithm == 'BiGRU_2'):
                Y_pred = BiGRU_2_Algo(X_train,Y_train,X_test)
                
            if(algorithm == 'RNN'):
                Y_pred = RNN_Algo(X_train,Y_train,X_test)
            if(algorithm == 'RNN_2'):
                Y_pred = RNN_2_Algo(X_train,Y_train,X_test)
            if(algorithm == 'BiRNN'):
                Y_pred = BiRNN_Algo(X_train,Y_train,X_test)
            if(algorithm == 'BiRNN_2'):
                Y_pred = BiRNN_2_Algo(X_train,Y_train,X_test)
        
        
        Y_pred = Y_pred.reshape((-1,Y_test.shape[1]))
        Y_pred = np.array(Y_pred)
        
        print(Y_pred.shape)
        
        #print("Y_pred:\n\n",Y_pred)
        #print("Y_test:\n\n",Y_test)
        
        
        
        #correct = sum(np.sum((Y_pred == Y_test),axis=1) == Y_test.shape[1]) 
        
        # Append the Accuracy        
        #acc.append((correct/len(Y_test))*100)
        
        #print('Acc = ',(correct/len(Y_test))*100)
        
        from sklearn.metrics import accuracy_score
        
        acc.append(accuracy_score(Y_pred, Y_test)*100)
        print("Test set accuracy: {:.2f}".format(accuracy_score(Y_pred, Y_test)))
        
    return acc


def dataset_format_generator(X,Y,min_index,max_index,batch_size,lookback,step,delay,is_batch = True):
    
    import numpy as np
    
    total_size = (max_index - min_index)

    if ((total_size - lookback) % batch_size == 0):
        n_batch = ((total_size - lookback) / batch_size)
    else:
        n_batch = np.floor((total_size - lookback) / batch_size) + 1
    n_batch = int(n_batch)


    # Initializing Samples and Corresponding Targets
    samples = []
    targets = []

    for i in range(n_batch):

        start_index = (min_index + lookback + i*batch_size)
        end_index = min((min_index + lookback + (i+1)*batch_size), max_index)

        batch_indices = np.arange(start_index, end_index)

        batch = []
        target = []
        for j in batch_indices:
            row_indices = range(j - lookback, j, step)
            #target_indices = range(j - lookback + delay, j + delay, step)

            batch.append(X[row_indices])
            target.append(Y[j + delay])

        batch = np.array(batch)
        target = np.array(target).reshape(-1,Y.shape[1])

        samples.append(batch)
        targets.append(target)

    samples = np.array(samples)
    targets = np.array(targets)
    
    
    if(is_batch == False):
        X_format_dataset = samples[0]
        Y_format_dataset = targets[0]
        for i in range(1,samples.shape[0]):
            X_format_dataset = np.vstack((X_format_dataset,samples[i]))
            Y_format_dataset = np.vstack((Y_format_dataset,targets[i]))
            
        return X_format_dataset,Y_format_dataset
        
        
    return samples,targets
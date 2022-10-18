def Run_Algos(X,Y,n_cluster,path_savefig,metric1 = False,metric2 = True):

    
    ####################### Import Necessary Libraries #######################
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    
    
    ######################### Define Parameters ############################
    
    path = 'C:/Users/prash/Downloads/STOCK MARKET/'
    
    # Metric 1 Parameters
    period = 1200
    step = 200
    algo = 'XGBoost'
    
    # Metric 2 Parameters
    low = 70
    high = 90
    
    ######################## Import necessary Functions ########################
    
    os.chdir(path + 'CODE/ALGORITHMS/')
    from Model_Training_Functions import Metric_1_model,Metric_2_model
    
    
    path = path_savefig
    os.chdir(path)
    
    ##################### Importing Data ########################
    
    #X = pd.read_csv(path + 'DATASETS/' + 'X_2019.csv')
    #Y = pd.read_csv(path + 'DATASETS/' + 'Y_2019.csv')
    #encoded_Y = pd.read_csv(path + 'DATASETS/' + 'encoded_Y_2019.csv')
    
    ##################### Data Cleaning #########################
    
    X_stock = X[:,0:6]
    
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    encoded_Y = np_utils.to_categorical(encoded_Y)
    
    encoded_Y = np.array(encoded_Y)
    
    Y = np.array(Y)
    Y = Y.reshape(Y.size,1)
    
    acc = {'Metric 1 Combined' : [],
           'Metric 1 Stock' : [],
           'Metric 2 Combined' : [],
           'Metric 2 Stock' : [],}
    
    #days_test = 20
    
    ######################### Metric 1 ################################
    if(metric1==True):
        for days_test in range(1,11,1):
            
            print("***************** ",days_test," ******************")
            
            # Combined Dataset
            acc_combined_dataset_metric_1 = Metric_1_model(X,encoded_Y,
                                                           algorithm = algo,
                                                           training_period = period,
                                                           step = step,
                                                           days_test = days_test)
            
            
            acc['Metric 1 Combined'].append(acc_combined_dataset_metric_1)
            
            print('Combined Dataset Predicted !!!')
            print('The Accuracies Obtained are:\n')
            print(acc_combined_dataset_metric_1)
            
            # Stock Dataset
            acc_stock_dataset_metric_1 = Metric_1_model(X_stock,encoded_Y,
                                                        algorithm = algo,
                                                        training_period = period,
                                                        step = step,
                                                        days_test = days_test)
            
            
            #acc_news_dataset_metric_1 = Metric_1_model(X[:,7:],Y,
            #                                           algorithm = algo,
            #                                           training_period = period,
            #                                           step = step,
            #                                           days_test = days_test)
            
            acc['Metric 1 Stock'].append(acc_stock_dataset_metric_1)
            
            print('\nStock Dataset Predicted !!!')
            print('The Accuracies Obtained are:\n')
            print(acc_stock_dataset_metric_1)
            
            
            max_training_period = min(2300,len(X)-days_test)
            
            # Plotting the Accuracies Obtained
            plt.figure()
            plt.plot(range(period,max_training_period,step),acc_combined_dataset_metric_1,c='red',label = 'Stock + News')    
            plt.plot(range(period,max_training_period,step),acc_stock_dataset_metric_1,c='blue',label = 'Stock')
            #plt.plot(range(period,max_training_period,step),acc_news_dataset_metric_1,c='green',label = 'News')
            plt.legend(loc="upper left")
            plt.ylabel('Accuracy')
            plt.xlabel('Number of Days Trained')
            plt.title('ROLLING ACC FOR DIFFERENT DAYS')
            #plt.figure(figsize=(40, 40))
            plt.savefig(path + str(n_cluster) + '_Clusters_Rolling_Acc_' + str(days_test) + '_days_later.png')
            plt.show()
    
    ########################### Metric 2 #################################
    
    if(metric2==True):
    
        # Combined Dataset
        acc_combined_dataset_metric_2 = Metric_2_model(X,encoded_Y,
                                                       algorithm = algo,
                                                       low = low,
                                                       high = high)
        
        acc['Metric 2 Combined'].append(acc_combined_dataset_metric_2)
        # Stock Dataset
        acc_stock_dataset_metric_2 = Metric_2_model(X_stock,encoded_Y,
                                                    algorithm = algo,
                                                    low = low,
                                                    high = high)
        
        acc['Metric 2 Stock'].append(acc_stock_dataset_metric_2)
        #acc_news_dataset_metric_2 = Metric_2_model(X[:,7:],Y,
        #                                           algorithm = algo,
        #                                           low = low,
        #                                           high = high )
        
        
        # Plotting the Accuracies Obtained
        plt.figure()
        plt.plot(range(low,high,1),acc_combined_dataset_metric_2,c='red',label = 'Stock + News')    
        plt.plot(range(low,high,1),acc_stock_dataset_metric_2,c='blue',label = 'Stock')
        #plt.plot(range(low,high,1),acc_news_dataset_metric_2,c='green',label = 'News')
        plt.legend(loc="upper right")
        plt.ylabel('Accuracy')
        plt.xlabel('Percentage of Data Trained')
        plt.title('ACC FOR DIFFERENT % OF DATA TRAINED')
        #plt.figure(figsize=(40, 40))
        plt.savefig(path + str(n_cluster) + '_Grid_Search_Acc.png')
        plt.show()
    
    
    return acc
    
    ####################################### END ############################################
    

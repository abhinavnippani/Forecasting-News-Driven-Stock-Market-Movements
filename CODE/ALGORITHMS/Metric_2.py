########################## Importing Necessary Libraries ##############################

import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import os

######################## Defining Parameters #####################################

path = 'C:/Users/prash/Downloads/STOCK MARKET/'


low = 60
high = 85
#algo = ['XGBoost','RandomForest']
#algo = ['LSTM','LSTM_2']
#algo = ['BiLSTM','BiLSTM_2']
#algo = ['GRU','GRU_2']
#algo = ['BiGRU','BiGRU_2']
#algo = ['RNN','RNN_2']
#algo = ['BiRNN','BiRNN_2']

#algo = ['LinearSVM','KernelSVM']
algo = ['DecisionTree']


# How may Timesteps back the Input data should go
#lookback_list = np.arange(180,300,50)
lookback_list = np.arange(180,190,50)
# The Period(in Timesteps) in which the data is sampled
step = 1
# How many Timesteps in the Future the Targets should be 
delay = 1
batch_size = 128

########################## Import Files ##########################################

input_df = pd.read_csv(path + 'DATASETS/' + 'Input.csv')

output_df = pd.read_csv(path + 'DATASETS/' + 'Output.csv')

print('\n\nImported Files!!!\n')

########################## Data Cleaning ##########################################

dates = np.array(input_df.iloc[:,1])

X = np.array(input_df.iloc[:,2:])
output_df = np.array(output_df.iloc[:,1])


Y = np.array([1 if (dummy >=1) else 0 for dummy in output_df])


#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
encoded_Y = np_utils.to_categorical(Y)

encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])


X_stock = X[:,:6]



####################################################################################

os.chdir(path + 'CODE/ALGORITHMS/')
from Metrics import Metric_2_model,dataset_format_generator


#######################################################################################


import tensorflow as tf

acc = {}
for lookback in lookback_list:

    print("\n\n              ****** " + str(lookback) + " *****")
        
    for algo_name in algo:
    
        print("\n\n              ****** " + algo_name + " *****")
    
            
        X_formatted = X
        X_formatted_stock = X_stock
        Y_formatted = encoded_Y
            
            
        tf.keras.backend.clear_session()
        # Combined Dataset
        acc_combined_dataset_metric_2 = Metric_2_model(X_formatted,Y_formatted,
                                                       algorithm = algo_name,
                                                       low = low,
                                                       high = high,
                                                       lookback = lookback)
        print("\n\nAccuracies of Combined Dataset for " + algo_name + " are:\n\n",acc_combined_dataset_metric_2)
        
        #tf.keras.backend.clear_session()
        
        # Stock Dataset
        acc_stock_dataset_metric_2 = Metric_2_model(X_formatted_stock,Y_formatted,
                                                    algorithm = algo_name,
                                                    low = low,
                                                    high = high,
                                                    lookback = lookback)
        
        print("\n\nAccuracies of Stock Dataset for " + algo_name + " are:\n\n",acc_stock_dataset_metric_2)
    
        acc[algo_name + ' Combined Accuracy'] = acc_combined_dataset_metric_2
        acc[algo_name + ' Stock Accuracy'] = acc_combined_dataset_metric_2
    
        
        # Plotting the Accuracies Obtained
        plt.figure()
        plt.plot(range(low,high,1),acc_combined_dataset_metric_2,c='red',label = 'Stock + News')    
        plt.plot(range(low,high,1),acc_stock_dataset_metric_2,c='blue',label = 'Stock')
        plt.legend(loc="lower left")
        plt.ylabel('Accuracy')
        plt.xlabel('Percentage of Data Trained')
        plt.title(algo_name + ' Train Test Split')
        #plt.figure(figsize=(40, 40))
        #plt.savefig(path + '/CODE/ALGORITHMS/' + algo_name + '_' + str(lookback) + '_Grid_Search_Acc.png')
        plt.savefig(path + '/CODE/ALGORITHMS/' + algo_name + '_Grid_Search_Acc.png')
        plt.show()
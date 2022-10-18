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


low = 70
high = 90
#algo = ['XGBoost','RandomForest','LSTM']
algo = ['DecisionTree']



########################## Import Files ##########################################

input_df = pd.read_csv(path + 'DATASETS/' + 'Input.csv')

output_df = pd.read_csv(path + 'DATASETS/' + 'Output.csv')

print('\n\nImported Files!!!\n')

########################## Data Cleaning ##########################################

dates = np.array(input_df.iloc[:,1])

X = np.array(input_df.iloc[:,2:])
output_df = np.array(output_df.iloc[:,1])


Y = np.array([1 if (dummy >=1) else 0 for dummy in output_df])


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
encoded_Y = np_utils.to_categorical(Y)

encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])


X_stock = X[:,:6]

##################################################################

os.chdir(path + 'CODE/ALGORITHMS/')
from Metrics import Metric_1_model,dataset_format_generator


#######################################################################################

# Metric 1 Parameters
period = 60
step = 60

days_test = 1


#np.eye(2)[[[0],[1],[1]]]

max_training_period = min(2000,len(X)-days_test)

acc = {}

for algo_name in algo:
    
    print("\n\n              ****** " + algo_name + " *****")
    
    
    X_formatted = X
    X_formatted_stock = X_stock
    Y_formatted = encoded_Y
    
        
        
    
    # Combined Dataset
    acc_combined_dataset_metric_1 = Metric_1_model(X_formatted,Y_formatted,max_training_period,
                                                   algorithm = algo_name,
                                                   training_period = period,
                                                   step_model = step,
                                                   days_test = 1)
    print("\n\nAccuracies of Combined Dataset for " + algo_name + " are:\n\n",acc_combined_dataset_metric_1,"\n")
    
    # Stock Dataset
    acc_stock_dataset_metric_1 = Metric_1_model(X_formatted_stock,Y_formatted,max_training_period,
                                                algorithm = algo_name,
                                                training_period = period,
                                                step_model = step,
                                                days_test = 1)
    
    print("\n\nAccuracies of Stock Dataset for " + algo_name + " are:\n\n",acc_stock_dataset_metric_1,"\n")

    acc[algo_name + ' Combined Accuracy'] = acc_combined_dataset_metric_1
    acc[algo_name + ' Stock Accuracy'] = acc_combined_dataset_metric_1

    
    # Plotting the Accuracies Obtained
    plt.figure()
    plt.plot(range(period,max_training_period,step),acc_combined_dataset_metric_1,c='red',label = 'Stock + News')    
    plt.plot(range(period,max_training_period,step),acc_stock_dataset_metric_1,c='blue',label = 'Stock')
    plt.legend(loc="lower left")
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Days Trained')
    plt.title(algo_name + ' Window Based Roll Forward Partition')
    #plt.title(algo_name + ' Roll_Forward_Partition')
    #plt.figure(figsize=(40, 40))
    #plt.savefig(path + '/CODE/ALGORITHMS/' + algo_name + '_Roll_Forward_Partition_Acc.png')
    plt.savefig(path + '/CODE/ALGORITHMS/' + algo_name + '_Window_Training_Acc.png')
    plt.show()
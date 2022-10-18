
########################## Importing Necessary Libraries ##############################

import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np

######################## Defining Parameters #####################################

path = 'C:/Users/prash/Downloads/STOCK MARKET/'

########################## Import Files ##########################################

stock_data = pd.read_excel(path + 'DATASETS/' + 'Stock_data.xlsx')

news_data = pd.read_csv(path + 'DATASETS/' + 'preprocessed_news_df_2001_2018.csv')

print('\n\nImported Files!!!\n')

########################## Data Cleaning #########################################

# Convert the Stock data into % Changes
for i in range(stock_data.shape[0]-1):

    stock_data.iloc[i+1,1:] = (stock_data.iloc[i+1,1:] - stock_data.iloc[i,1:])/stock_data.iloc[i,1:]

stock_data = stock_data.iloc[1:,:]

news_data = news_data.iloc[:,1:]

print('Converted Stock Data Values into Returns!!!\n')

###################### Convert Date Columns into same format ##########################

# Stock
for i in range(stock_data.shape[0]):
    dummy = datetime.strptime(str(stock_data.iloc[i,0]), '%Y-%m-%d %H:%M:%S')
    stock_data.iloc[i,0] = dummy.strftime('%m/%d/%y')


# News
for i in range(news_data.shape[0]):
    dummy = datetime.strptime(str(int(news_data.iloc[i,-1])), '%Y%m%d')
    news_data.iloc[i,-1] = dummy.strftime('%m/%d/%y')
    
print('Converted Date Values into same Format!!!\n')

########################## Input Dataset ######################################

print('Creating Input Dataset:\n')

X = []
for i in range(stock_data.shape[0]-1):
    for j in range(news_data.shape[0]):
        if(news_data.iloc[j,-1]==stock_data.iloc[i,0]):
            dummy = list(stock_data.iloc[i,:]) + list(news_data.iloc[j,:-1])                      
            #Y.append(1 if (stock_data.iloc[i+1,2] >= stock_data.iloc[i,2]) else 0)            
            X.append(dummy)
    if(i%100 == 0):
        print(i)
                
X = pd.DataFrame(X)
#X.iloc[:,0] = pd.to_datetime(X.iloc[:,0])

print('\nCreated Input Dataset!!!\n')

pd.DataFrame.to_csv(X,path + 'DATASETS/' + 'Input.csv') 


######################### Output Dataset ######################################

print('Creating Output Dataset:\n')

Y = []
output = []
for i in range(X.shape[0]):
    for j in range(stock_data.shape[0]):
        if(X.iloc[i,0]==stock_data.iloc[j,0]):
            dummy = (((stock_data.iloc[j+1,5] - stock_data.iloc[j,5])/stock_data.iloc[j,5]) * 100)
            output.append(dummy)
            Y.append(1 if (dummy >=1) else 0)
    if(i%100 == 0):
        print(i)
                    
output = np.array(output)
output = output.reshape((-1,1))   
output = pd.DataFrame(output) 

print('\nCreated Output Dataset!!!\n')

pd.DataFrame.to_csv(output,path + 'DATASETS/' + 'Output.csv') 

############################################################################

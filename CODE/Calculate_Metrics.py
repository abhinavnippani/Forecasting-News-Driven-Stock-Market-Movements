
################################### Import Necessary Libraries ###################################

import os
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

################################ Defining Parameters ###########################################

path = 'C:/Users/prash/Downloads/STOCK MARKET/'

clusters = list(np.arange(2, 6, 1))

############################### Calling Other Python Scripts ##############################

# Combine News ans Stock Datasets
os.chdir(path + 'CODE/')
from Combining_Datasets import *

X = np.array(X.iloc[:,1:])








Y_val


# Function for Generating Accuracies
os.chdir(path + 'CODE/ALGORITHMS/')
from Run_Algos import Run_Algos

############################# Direction Prediction ###################################

path_images = path + 'OUTPUT/LDA_Gensim_Sentiment_Tfidf_2010_2018/'
os.makedirs(path_images + 'Direction/', exist_ok=True)
    
acc_direction = Run_Algos(X,Y,2,path_images + 'Direction/',True,True)

########################### MultiClass Prediction ######################################

kmeans_params = {'Clusters': [],
                 'Silhouette Score': [] }

for n_cluster in clusters:
    
    print("********************* N_Cat = " + str(n_cluster) + " **********")
    
    kmeans = KMeans(n_clusters=n_cluster).fit(output)
    Y = kmeans.predict(output)
    
    os.makedirs(path_images + str(n_cluster) + '_Clusters/', exist_ok=True)
    
    Run_Algos(X,Y,n_cluster,path_images + str(n_cluster) + '_Clusters/')
    
    label = kmeans.labels_
    sil_coeff = silhouette_score(output, label, metric='euclidean')
    
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))




for n_cluster in clusters:
    
    print("********************* N_Cat = " + str(n_cluster) + " **********")
    
    kmeans = KMeans(n_clusters=n_cluster).fit(output)
    Y = kmeans.predict(output)
    
    os.makedirs(path_images + str(n_cluster) + '_Clusters/', exist_ok=True)
    
    Run_Algos(X,Y,n_cluster,path_images + str(n_cluster) + '_Clusters/')
    
    label = kmeans.labels_
    sil_coeff = silhouette_score(output, label, metric='euclidean')
    
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))



import matplotlib.pyplot as plt


kmeans = KMeans(n_clusters=3).fit(output)
Y = kmeans.predict(output)


plt.figure(figsize=(8, 6))
plt.scatter(np.arange(0,len(output)),output, c=kmeans.labels_.astype(float))

###########################################################################################











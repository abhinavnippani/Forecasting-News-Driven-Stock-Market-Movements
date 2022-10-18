############################ Importing Libraries #################################

import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

############################## Parameters ################################

# Defining the path
path = 'C:/Users/prash/Downloads/STOCK MARKET/'

# LDA Parameters
n_categ_list = list(np.arange(5, 16, 3))

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.4))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.4))
beta.append('symmetric')

######################### Importing Necessary Functions ######################

#Importing necessary functions
os.chdir(path + 'CODE/NEWS/')
from News_PreProcessing_Functions import *

########################## Importing Data ###################################

stock_data = pd.read_excel(path + 'Stock_data.xlsx')

#news_df = pd.read_csv(path + 'DATASETS/' + 'news_df_2019.csv')
news_df = pd.read_csv(path + 'DATASETS/' + 'india-news-headlines.csv')

########################## Data Cleaning ####################################

# Removing duplicated rows/news headlines
news_df = news_df.drop_duplicates()
# Rearranging columns to the correct format
#news_df = news_df.iloc[:,[1,2,3,4]]
# Renaming the column names
#news_df.columns = ['Date', 'Headline', 'Alternate Headline','Link']
news_df.columns = ['Date', 'Category' ,'Headline']

#Combining the News Headlines and Alternate Headlines into one column
#data = news_df['Headline'].str.cat(news_df['Alternate Headline'])

data = news_df['Headline']

#Remove all NAs from the dataset
data = data.dropna()
data = list(data)

######################## Text Cleaning ##################################

# Initialize a pre-defined stopwords list
stopwords = set(STOPWORDS)

# Clean the text by callng the 'text_cleaning' function
data = text_cleaning(data,stopwords,path)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
response = tfidf.fit_transform(data)
feature_names = tfidf.get_feature_names()

tfidf_result = {'Word' : [],
                'Tfidf_Value' : []}

for col in response.nonzero()[1]:
    tfidf_result['Word'].append(feature_names[col])
    tfidf_result['Tfidf_Value'].append(response[0, col])
    #print (feature_names[col], ' - ', response[0, col])

tfidf_result = pd.DataFrame(tfidf_result)

stopword_list = list(tfidf_result[tfidf_result['Tfidf_Value'] <= 0]['Word'])
# Add a list of stopwords
stopwords.update(stopword_list)

len(stopwords)
    
# Clean the text by callng the 'text_cleaning' function
data = text_cleaning(data,stopwords,path)

############################ Create Corpus #########################################

import gensim.corpora as corpora

# Split each sentence into a list of words    
df1 = [sent.split() for sent in data]
texts = df1

# Create Dictionary of all the words
id2word = corpora.Dictionary(df1)

# Create Corpus(Term Document Frequency)
corpus = [id2word.doc2bow(text) for text in texts]

############################# HyperParameter Tuning of LDA Parameters #############################

model_results = Tuning_LDA_Params(texts = texts,
                                  corpus = corpus,
                                  id2word = id2word,
                                  n_categ_list = n_categ_list,
                                  alpha = alpha,
                                  beta = beta)


pd.DataFrame.to_csv(model_results,path + 'DATASETS/' + 'Coherence_Scores.csv') 
    

############################## Creating the Optimal LDA Model ###############################

best_model_params = model_results.iloc[model_results['Coherence'].idxmax(),:]

k = int(best_model_params['Topics']) #15
#k=25

a = best_model_params['Alpha'] #0.81
#a='asymmetric'

b = best_model_params['Beta'] #0.01
#b=0.31

optimal_model,coherence_score = LDA_Gensim(df = texts,
                                             corpus = corpus,
                                             id2word = id2word,
                                             num_topics = k,
                                             alpha = a,
                                             beta = b)

print('\nFinal Number of Categories: ', k)
print('\nFinal Coherence Score: ', coherence_score)


############################ Sentiment Analysis ###########################

sentiments = Sentiment_Analysis(data)

########################### Create Final PreProcessed DataFrame ##############################

all_topics = optimal_model.get_document_topics(corpus, per_word_topics=True)


preprocessed_news_df = Create_Preprocessed_df(all_topics,news_df,sentiments,
                                              k = k,
                                              is_sentiment = True)



preprocessed_news_df = preprocessed_news_df[preprocessed_news_df.iloc[:,k].notna()]
    
pd.DataFrame.to_csv(preprocessed_news_df,path + 'DATASETS/' + 'preprocessed_news_df_2001_2018.csv') 
    
    
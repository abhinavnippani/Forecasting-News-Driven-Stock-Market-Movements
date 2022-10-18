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
lda_limit=30; lda_start=5; lda_step=5;

######################### Importing Necessary Functions ######################

#Importing necessary functions
os.chdir(path + 'CODE/NEWS/')
from News_PreProcessing_Functions import text_cleaning
from News_PreProcessing_Functions import compute_coherence_values
from News_PreProcessing_Functions import naive_bayes_algo
from News_PreProcessing_Functions import LDA_Mallet_Model
                                         
########################## Importing Data ###################################

stock_data = pd.read_excel(path + 'Stock_data.xlsx')

news_df = pd.read_csv(path + 'DATASETS/' + 'news_df_2019.csv')

#news_data = pd.read_csv(path + 'CODE/' + 'news_dataset.csv')
#news_data1 = pd.read_csv(path + 'CODE/' + 'news_dataset1.csv')
#news_date_link = pd.read_csv(path + 'CODE/' + 'news_date_link.csv')

########################## Data Cleaning ####################################

# Appending the news datasets
#news_data = news_data.append(news_data1)

# Finding the Dates for the News Headlines
#df = pd.merge(news_date_link, news_data.iloc[:, 2:5], how='left', on='Link')
# Removing duplicated rows/news headlines
news_df = news_df.drop_duplicates()
# Rearranging columns to the correct format
news_df = news_df.iloc[:,[1,2,3,4]]
# Renaming the column names
news_df.columns = ['Date', 'Headline', 'Alternate Headline','Link']

#Combining the News Headlines and Alternate Headlines into one column
data = news_df['Headline'].str.cat(news_df['Alternate Headline'])
#Remove all NAs from the dataset
data = data.dropna()
data = list(data)

######################## Text Cleaning ##################################

# Initialize a pre-defined stopwords list
stopwords = set(STOPWORDS)
# Add a list of stopwords
stopwords.update(["per", "cent", "crore", "rs", "india",
                  "say","says","year","may","three","million",
                 "us","will","indian","said","new","indias"])
  

# Clean the text by callng the 'text_cleaning' function
data = text_cleaning(data,stopwords)


############################ Sentiment Analysis ###########################

sentiments = []

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
for i in range(len(data)):
    sentiments.append(sia.polarity_scores(data[i])['compound'])

############################ LDA #########################################

import gensim.corpora as corpora
    
######## Create Corpus ###########

# Split each sentence into a list of words    
df1 = [sent.split() for sent in data]
texts = df1

# Create Dictionary of all the words
id2word = corpora.Dictionary(df1)

# Create Corpus(Term Document Frequency)
corpus = [id2word.doc2bow(text) for text in texts]


# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
#alpha.append('symmetric')
#alpha.append('asymmetric')


# Beta parameter
#beta = list(np.arange(0.01, 1, 0.3))
#beta.append('symmetric')

model_results = {'Topics': [],
                 'Alpha': [],
                 'Coherence': []
                }

# iterate through validation corpuses

for k in range(lda_start,lda_limit,lda_step):
    # iterate through alpha values
    for a in alpha:
        # iterare through beta values
        #for b in beta:
        print("Number of Topics: ",k)
        print("Alpha: ",a)
        #print("Beta: ",b)
        
        # get the coherence score for the given parameters
        model,coherence_score = LDA_Mallet_Model(df = texts,
                                             corpus = corpus,
                                             num_topics = k,
                                             id2word = id2word,
                                             alpha = a)
        # Save the model results
        model_results['Topics'].append(k)
        model_results['Alpha'].append(a)
        #model_results['Beta'].append(b)
        model_results['Coherence'].append(coherence_score)
        
        print("Coherence Score: ",coherence_score)
        print("********************************")


model_results = pd.DataFrame(model_results)

best_model_params = model_results.iloc[model_results['Coherence'].idxmax(),:]

############################################################

k = int(best_model_params['Topics']) #15

a = best_model_params['Alpha'] #0.31

final_lda_model,coherence_score = LDA_Mallet_Model(df = df1,
                                                   corpus = corpus,
                                                   num_topics = k,
                                                   id2word = id2word,
                                                   alpha = a)
print('\nFinal Number of Categories: ', k)
print('\nFinal Coherence Score: ', coherence_score)

############ Getting the Keywords in Each Category #########

n_categ = k

# Getting the keywords and respective importance of each keyword in the category
result = final_lda_model.show_topics(num_topics = n_categ, 
                                     num_words = 200, 
                                     formatted = False)

#Creating a list of Keywords for each category
keywords_list = []
for i in range(n_categ):
    keywords_list.append(list(dict(result[i][1]).keys()))

# Can take a long time to run.
#n_categ,keywords_list = compute_coherence_values(data = data,
#                                          start = lda_start, 
#                                          limit = lda_limit, 
#                                          step = lda_step)

############################ Naive Bayes ####################################

logprior,loglike = naive_bayes_algo(keywords_list,data,n_categ)

############### Identifying the LogLikelihood Values for each Category of each News ############# 

news_categories_df = []
# Iterating over News
for row_news in np.array(news_df):
    #Combining Headline and Alternate Headline
    news = [row_news[1] + row_news[2]]
    # Proceed if News is not NA
    if(pd.isnull(news[0]) == False):
        #Cleaning the text and splitting it
        cleaned_news = text_cleaning(news,stopwords)[0].split()
        #Setting the Default value of Loglikelihood values to LogPrior calculated before
        row_news_categories = pd.DataFrame(logprior+[row_news[0]])
        #Finding the Loglikelihood for each category
        for i in range(n_categ):
            for word in cleaned_news:
                row_news_categories[0][i] += loglike[i].get(word,0)
        # Append the row wise news categories
        news_categories_df.append(row_news_categories[0])

news_categories_df = np.array(news_categories_df)


zzz = pd.DataFrame(news_categories_df)

################## Identifying the Category and corresponding Date of News ###################

date_news_category_df = []
# Iterating over Dates
for day in list(set(news_categories_df[:,(news_categories_df.shape[1]-1)])):    
    categories = []    
    for i in range(news_categories_df.shape[0]):
        # If the dates match,
        if(news_categories_df[i,news_categories_df.shape[1]-1] == day):
            # Append the likelihood values
            categories.append((list(news_categories_df[i,0:(news_categories_df.shape[1]-1)])))    
    categories = np.array(categories)    
    # Take the Index of Maximum Likelihood Value as the Category Number
    for i in range(categories.shape[0]):
        date_news_category_df.append([np.argmax(np.array(categories[i,0:(news_categories_df.shape[1]-1)])),
                                      day])
        #date_news_category_df.append([(np.array(categories[i,0:(news_categories_df.shape[1]-1)])),
        #                              day])        

date_news_category_df = pd.DataFrame(date_news_category_df)

####################### Forming the Encoded News Dataframe ###########################

encoded = to_categorical(date_news_category_df.iloc[:,0])


for i in range(len(sentiments)):
    if(i%1000 == 0):
        print(i)
    encoded[i,:] = encoded[i,:]*sentiments[i]
    

encoded_categories_df = pd.concat((pd.DataFrame(encoded), date_news_category_df.iloc[:,1]), axis=1)


####################### Forming the Final PreProcessed News DataFrame ######################

preprocessed_news_df = []
# Iterating over Dates
for day in list(set(encoded_categories_df.iloc[:,-1])):
    # Calculating the number of columns and rows
    columns = encoded_categories_df.shape[1]
    rows = encoded_categories_df.shape[0]
    # Taking the average number of News Categories in a Day
    preprocessed_news_date = list(encoded_categories_df.iloc[:,0:columns-1][encoded_categories_df.iloc[:,columns-1]==day].sum(axis=0)/
                                  rows)
    # Appending the Date
    preprocessed_news_date.append(day)
    # Appending each row    
    preprocessed_news_df.append(preprocessed_news_date)

# Converting to DataFrame        
preprocessed_news_df = pd.DataFrame(preprocessed_news_df)

############################ Saving the Final PreProcessed DataFrame ######################

pd.DataFrame.to_csv(preprocessed_news_df,path + 'DATASETS/' + 'preprocessed_news_df_2019.csv')


############################################ END ##########################################################

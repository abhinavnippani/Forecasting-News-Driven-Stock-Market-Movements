
############################# Text Cleaning ####################################

def text_cleaning(data,stopwords,path):
    
    ########## Import Libraries ########
    
    import pandas as pd
    import re
    from collections import Counter
    
    ########## Text Cleaning ##########
    
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    print('New Line Characters Removed !!!')
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    print('Distracting Single Quotes Removed !!!')
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    print('Emails Removed !!!')
    # Remove Punctuation
    data = [re.sub(r'[^\w\s]',' ', sent) for sent in data]
    print('Punctuations Removed !!!')
    # Remove Digits/Numericals
    data = [re.sub(" \d+", ' ', sent) for sent in data]
    print('Digits Removed !!!')
    # Remove Extra Whitespace
    data = [re.sub(' +', ' ', sent) for sent in data]
    print('Whitespace Removed !!!')
    # Convert to Lowercase
    data = [sent.lower() for sent in data]
    print('Converted to Lowercase !!!')    
    # Remove Stopwords
    stopwords_dict = Counter(stopwords)
    for i in range(len(data)):
        data[i] = ' '.join([word for word in data[i].split() if word not in stopwords_dict])
        if(i%10000 == 0):
            print(i)
    
    data = pd.DataFrame(data)
    data = data.iloc[:,0]
    
    
    
    #############
    
    return data


########################################### LDA #########################################

def LDA_Mallet_Model(df,corpus,num_topics,id2word,alpha):
    
    ######### Import Libraries #########

    from gensim.models.wrappers import LdaMallet
    import os
    from gensim.models import CoherenceModel
    
    ######## Parameters #########
    
    mallet_path = 'C:\\Users\\prash\\Downloads\\mallet-2.0.8\\bin\\mallet'
    os.environ['MALLET_HOME'] = 'C:\\Users\\prash\\Downloads\\mallet-2.0.8\\'
    
    ############################
    
    # Creating the model
    ldamallet_model = LdaMallet(mallet_path, 
                          corpus = corpus, 
                          num_topics = num_topics, 
                          id2word = id2word,
                          alpha = alpha)
    # Computing Coherence Score (The Higher the value, the better the Topics Generated)
    coherence_model_ldamallet = CoherenceModel(model = ldamallet_model, 
                                               texts = df, 
                                               dictionary = id2word, 
                                               coherence = 'c_v')
    
    coherence_score = coherence_model_ldamallet.get_coherence()
    
    ################################
    
    return ldamallet_model,coherence_score




def LDA_Gensim(df,corpus,id2word,num_topics,alpha,beta):

    from gensim.models import CoherenceModel,ldamodel
    
    # Build LDA model
    lda_model = ldamodel.LdaModel(corpus = corpus,
                                   id2word = id2word,
                                   num_topics = num_topics, 
                                   random_state=100,
                                   alpha=alpha,
                                   eta = beta,
                                   per_word_topics=True)
    
    
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=df, 
                                         dictionary=id2word, 
                                         coherence='c_v')
    
    coherence_score = coherence_model_lda.get_coherence()
    
    
    return lda_model,coherence_score


############################## Choose Best LDA Model ###################################

def compute_coherence_values(data, limit, start=2, step=3):
    
    ######### Import Libraries #########
    
    import numpy as np
    import matplotlib.pyplot as plt
    import gensim.corpora as corpora
    
    ######## Create Corpus ###########
    
    # Split each sentence into a list of words    
    df1 = [sent.split() for sent in data]
    texts = df1

    # Create Dictionary of all the words
    id2word = corpora.Dictionary(df1)
    
    # Create Corpus(Term Document Frequency)
    corpus = [id2word.doc2bow(text) for text in texts]
    
    ######### Calculate Coherence Values ##########
    
    # Initialize empty lists
    coherence_values = []
    model_list = []
    no_topics = []
    # Iterate over number of categories  
    for num_topics in range(start, limit, step):
        # Create a LDA Model and compute Coherence Score
        model,coherence_score = LDA_Mallet_Model(df = texts,
                                                 corpus = corpus,
                                                 num_topics = num_topics,
                                                 id2word = id2word)
        # Print the Number of Categories and its corresponding Coherence Score
        print([num_topics,coherence_score])
        # Append the model to corresponding list
        model_list.append(model)
        # Append Coherence Score calculated to the corresponding list
        coherence_values.append(coherence_score)
        # Append Number of Categories to its corresponding list 
        no_topics.append(num_topics)
         
    ########### Plot the Obtained Coherence Values Vs Number of Categories ##########
        
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Categories")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    ############ Final LDA Model ##########
    
    # Setting the number of Categories to the Maximum Coherence Score obtained 
    n_categ = no_topics[np.argmax(coherence_values)]
    # Creating the Final Model
    final_lda_model,coherence_score = LDA_Mallet_Model(df = df1,
                                                       corpus = corpus,
                                                       num_topics = num_topics,
                                                       id2word = id2word)
    print('\nFinal Number of Categories: ', n_categ)
    print('\nFinal Coherence Score: ', coherence_score)
    
    ############ Getting the Keywords in Each Category #########
    
    # Getting the keywords and respective importance of each keyword in the category
    result = final_lda_model.show_topics(num_topics = n_categ, 
                                         num_words = 200, 
                                         formatted = False)
    
    #Creating a list of Keywords for each category
    keywords_list = []
    for i in range(n_categ):
        keywords_list.append(list(dict(result[i][1]).keys()))
        
    ###########################

    return n_categ,keywords_list


################################## Naive Bayes ####################################### 


def naive_bayes_algo(keywords_list,data,n_categ):
    
    import numpy as np
    
    text = ' '.join(sent for sent in data)
    
    # Define the Vocabulary of the Overall Dataset
    vocab_overall = dict()
    for word in text.split():
        vocab_overall[word] = vocab_overall.get(word,0)+1
    
        
    # Calculate Number of Words in each Category   
    length = []
    for i in range(n_categ):
        length.append(len(keywords_list[i]))   
    
        
    # Calculate LogPriors of each Category    
    logprior = []
    for i in range(n_categ):
        logprior.append(np.log(length[i]/len(text)))
        
    
    # Define the Vocabulary of each Category    
    vocab = []
    for i in range(n_categ):
        a = dict()
        for word in keywords_list[i]:
            a[word] = a.get(word,0)+1
        vocab.append(a)
        
    
    # Calculate LogLikelihoods of each Category     
    loglike = []
    for i in range(n_categ):
        a = dict()
        denom = 0
        for key in vocab_overall.keys():
            denom = denom + (vocab[i].get(key,0)+1)
        for key in vocab_overall.keys():
            a[key] = np.log((vocab[i].get(key,0)+1)/denom)
        loglike.append(a)
     
        
    return logprior,loglike


############################# Sentiment Analysis #############################################
    
def Sentiment_Analysis(data):

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for i in range(len(data)):
        sentiments.append(sia.polarity_scores(data[i])['compound'])
        if(i%10000 == 0):
            print(i)
        
    
    return sentiments

############################# HyperParameter Tuning of LDA Parameters #############################

def Tuning_LDA_Params(texts,corpus,id2word,n_categ_list,alpha,beta):
    
    import pandas as pd
    
    model_results = {'Topics': [],
                     'Alpha': [],
                     'Beta' : [],
                     'Coherence': []
                    }
    
    # iterate through validation corpuses
    
    for k in n_categ_list:
        # iterate through alpha values
        for a in alpha:
            # iterare through beta values
            for b in beta:
                print("********************************")
                print("Number of Topics: ",k)
                print("Alpha: ",a)
                print("Beta: ",b)
                
                # get the coherence score for the given parameters
                model,coherence_score = LDA_Gensim(df = texts,
                                                     corpus = corpus,
                                                     id2word = id2word,
                                                     num_topics = k,
                                                     alpha = a,
                                                     beta = b)
                # Save the model results
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(coherence_score)
                
                print("Coherence Score: ",coherence_score)
                  
    model_results = pd.DataFrame(model_results)
    
    return model_results

############################## Create Final PreProcessed Dataframe ####################

def Create_Preprocessed_df(all_topics,news_df,sentiments,k,is_sentiment = True):
    
    import numpy as np
    import pandas as pd
    
    x = 0
    topic_contribution_df = []
    for doc_topics, word_topics, phi_values in all_topics:
        topic_contribution_df.append(doc_topics)
        x += 1
        if(x%1000==0):
            print(x)
        
        
    final_df = np.zeros((news_df.shape[0],k))
    for i in range(news_df.shape[0]):
        for categ in dict(topic_contribution_df[i]).keys():
            final_df[i,categ] = dict(topic_contribution_df[i])[categ]
    
    print('Divided into Categories!!')
    
    if(is_sentiment == True):    
        for i in range(len(sentiments)):
            final_df[i,:] = final_df[i,:]*sentiments[i]
            
    final_df = pd.DataFrame(final_df)
        
    encoded_categories_df = pd.concat((final_df, news_df.iloc[:,0]), axis=1)
    
    print('Multiplied with Sentiments!!')
    
    preprocessed_news_df = []
    # Iterating over Dates
    for day in list(set(encoded_categories_df.iloc[:,-1])):
        
        # Calculating the number of columns and rows
        columns = encoded_categories_df.shape[1]
        rows = encoded_categories_df.shape[0]
        # Taking the average number of News Categories in a Day
        preprocessed_news_date = list(encoded_categories_df.iloc[:,0:columns-1][encoded_categories_df.iloc[:,columns-1]==day].sum(axis=0)/
                                      rows)
        print(day)
        # Appending the Date
        preprocessed_news_date.append(day)
        # Appending each row    
        preprocessed_news_df.append(preprocessed_news_date)
    
    # Converting to DataFrame        
    preprocessed_news_df = pd.DataFrame(preprocessed_news_df)
    
    
    return preprocessed_news_df

######################################################################################

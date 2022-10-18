import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib.request as urb
from urllib.error import HTTPError, URLError
import socket
import logging
import time
from datetime import datetime

path = 'C:/Users/prash/Downloads/STOCK MARKET/'

stock_data = pd.read_excel(path + 'Stock_data.xlsx')

days_completed = pd.read_csv(path + 'CODE/' + 'days_completed.csv')
days_completed = int(days_completed.iloc[0,1])

news_dataset = pd.read_csv(path + 'CODE/' + 'news_dataset.csv')
news_dataset = news_dataset.values.tolist()

date_list = []
for i in range(stock_data.shape[0] - 1):
    dummy = abs(((stock_data.iloc[i+1,5] - stock_data.iloc[i,5])/stock_data.iloc[i,5]) * 100)
    if(dummy >= 1):
        date_list.append(stock_data.iloc[i,0])
        
filtered_stock_dataset = stock_data.loc[stock_data.iloc[:,0].isin(date_list)]

url = 'https://www.business-standard.com/todays-paper'
driver = webdriver.Chrome(path + 'chromedriver')
driver.get(url)

headlines = []
alt_headlines = []
date = []
links = []

#news_dataset = []
for i in range(days_completed,len(date_list)):
    
    year = date_list[i].year
    month = date_list[i].month
    day = date_list[i].day
    
    yearstr = str(year)
    monthstr = str(month)
    daystr = str(day)
    
    #If the digit is single digit, convert to double digit by putting a '0' before it
    if (month < 10):
        monthstr = '0' + str(month)
    if (day < 10):
        daystr = '0' + str(day)

    print("****************    " + str(day) + "/" + str(month) + "/" + str(year) + "     ******************")
              
    #Checking if the Date exists or not
    t=0
    try:
        date = str(pd.to_datetime(str(year)+str(month)+str(day), format='%Y%m%d')).split()[0]
    except:
        t=1
 
    if(t==0):
        driver.get(url)
        date_element = driver.find_element_by_id('dateb')
        date_element.clear()
        date_element.send_keys(str(month) + "/"+ str(day) + "/" + str(year))
        #date_element.send_keys(monthstr + "/"+ daystr + "/" + yearstr)
        date_button = driver.find_element_by_xpath("//input[@class='sort-go-btn']")
        date_button.click()
        time.sleep(5)
        list_of_hrefs = []
        content_blocks = driver.find_elements_by_class_name("main-cont-left")
        
        for block in content_blocks:
            elements = block.find_elements_by_tag_name("a")
            for el in elements:
                if(el.get_attribute("href") != url):
                    list_of_hrefs.append(el.get_attribute("href"))

        for href in list_of_hrefs:
            print(href)
            #driver.get(href)
            try:
                html = urlopen(href, timeout=10)
            except HTTPError as error:
                logging.error('Data not retrieved because %s\nURL: %s', error, url)
            except URLError as error:
                if isinstance(error.reason, socket.timeout):
                    logging.error('socket timed out - URL %s', url)
                else: 
                    logging.error('some other error happened')
            else:
                logging.info('Access successful.')
                                  

            #html = urb.urlopen(href, timeout=20)
            soup = BeautifulSoup(html, "html.parser")

            date = str(pd.to_datetime(str(year)+str(month)+str(day), format='%Y%m%d')).split()[0]

            #headline = driver.find_element_by_xpath("//h1[@class='headline']")
            for item in soup.select(".headline"):
                headline = item.get_text()                 

            #alt_headline = driver.find_element_by_xpath("//h2[@class='alternativeHeadline']")
            alt_headline = " "
            for item in soup.select(".alternativeHeadline"):
                alt_headline = item.get_text() 
            
            link = href

            #news_dataset.append([date,headline.text,alt_headline.text,link])
            news_dataset.append([date,headline,alt_headline,link])

            news_df = pd.DataFrame(news_dataset)
            news_df = news_df.iloc[:,1:]
            news_df.columns = ["Date","Headline","Alternate Headline","Link"]

            pd.DataFrame.to_csv(news_df,path + "CODE/" + "news_dataset.csv")
            
            
            
    pd.DataFrame.to_csv(pd.DataFrame(np.array([i])),path + "CODE/" + "days_completed.csv")
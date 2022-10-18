# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:28:41 2019

@author: Vamsi.Naidu
"""

##importing and initializing variables
# =============================================================================
from lxml import html
from requests import get

import pandas as pd
import os

from data_prices import skus1mgDict
from datetime import datetime
 
from time import sleep

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.select import Select

pat = os.path.dirname(os.path.abspath(__file__))+"\\"

drivermg = webdriver.Chrome(r"C:\Users\Vamsi.naidu\Documents\chromedriver.exe")

# =============================================================================
# 1mg price gen
# =============================================================================

def mgPrice(drivermg,skus1mgDict,city):
    skus1mg = list(skus1mgDict.keys())
    df1mgAll = pd.DataFrame(columns=["Channel","Product","Price","Availability"])
    i=0
    drivermg.get("https://www.1mg.com/otc/"+str(skus1mg[0]))
    drivermg.find_element_by_class_name('styles__city-input___6e65P').clear()
    drivermg.find_element_by_class_name('styles__city-input___6e65P').send_keys(city)
    drivermg.find_element_by_class_name('LocationDropDown__city-item___XRtse').click()
    while i < len(skus1mg):
        sku = skus1mg[i]
        drivermg.get("https://www.1mg.com/otc/"+str(sku))
        product = skus1mgDict[sku]
        sleep(5)
        try:
            price = drivermg.find_element_by_class_name("PriceDetails__discount-div___nb724").text
        except NoSuchElementException:
            try:
                price = drivermg.find_element_by_class_name("SaleDetails__discount-price___3xUk9").text
            except NoSuchElementException:
                try:
                    price = drivermg.find_element_by_class_name("DrugPriceBox__best-price___32JXw").text
                except NoSuchElementException:
                    price = "Error "+product
        try:
            availability = drivermg.find_element_by_class_name("AvailableStatus__container___1R2Nk").text
            if "SOLD OUT" in availability:
                #price = "Out of Stock"
                stock = "Out of Stock"
            elif "ADD TO CART" in availability:
                stock = "In Stock"
        except NoSuchElementException:
            try:
                availability = drivermg.find_element_by_class_name("DrugAddToCart__add-to-cart___Qcxup").text
                if "ADD TO CART" in  availability:
                    stock = "In Stock"
                else:
                    stock = "Error "
            except NoSuchElementException:
                try:
                    availability = drivermg.find_element_by_class_name("AddToCart__add-to-cart___39skL").text
                    if "ADD TO CART" in availability:
                        stock = "In Stock"
                    else:
                        stock = "Error"
                except NoSuchElementException:
                    stock = "Error"
                
        price = price.split("₹")[1]
        print(price)
        if stock == "Out of Stock":
            df1mg = pd.DataFrame({"Channel":["1mg"],"Product":[product],"Price":[price],"Availability":[stock]})
            df1mgAll = pd.concat([df1mgAll,df1mg])
            print(price)
        i=i+1
    df1mgAll = df1mgAll[["Channel","Product","Price","Availability"]]
    return df1mgAll

mgCities = ['Delhi','Mumbai','Hyderabad','Bangalore','Chennai','Gurgaon']

#mgCities = ['Bangalore','Chennai','Gurgaon']

allCityDf = pd.DataFrame(['Channel','Product','Price','Avaialability','City'])

for city in mgCities:
    df = mgPrice(drivermg,skus1mgDict,city)
    df['City'] = city
    allCityDf = pd.concat([allCityDf,df],sort=False,ignore_index=True)

allCityDf.to_excel(pat+'mgPric.xlsx')

#Importing the libraries
from pymongo import MongoClient
import pandas as pd
#Client object
client = MongoClient()
#Mongo runs on port number 27017
client = MongoClient('localhost',27017)


#MGetting the database
db = client.Apple
#Creating a collection
stock_prices = db.stock_prices
#Getting the stock values
prices_apple = []
for x in stock_prices.find():
    prices_apple.append(x['stock_price'])
    
prices_apple = pd.DataFrame(prices_apple,columns=["AAPL"])

#getting the database
db = client.MSI
#Creating a collection
stock_prices = db.stock_prices
#Getting the stock values
prices_msi = []
for x in stock_prices.find():
    prices_msi.append(x['stock_price'])
    
prices_msi = pd.DataFrame(prices_msi,columns=["MSI"])

#getting the database
db = client.Starbucks
#Creating a collection
stock_prices = db.stock_prices
#Getting the stock values
prices_sbux = []
for x in stock_prices.find():
    prices_sbux.append(x['stock_price'])
    
prices_sbux = pd.DataFrame(prices_sbux,columns=["SBUX"])

stock_prices = pd.concat([prices_apple,prices_msi,prices_sbux],axis=1)
#Saving the dataframe
stock_prices.to_csv("aapl_msi_sbux.csv",index=False)



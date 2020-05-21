#Importing the libraries
from pymongo import MongoClient
import pandas as pd
from functools import reduce 

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
for x in stock_prices.find({'stock_type': 'COMMON_STOCK'}):
    prices_apple.append(x['stock_price']/72.3)
    
prices_apple = pd.DataFrame(prices_apple,columns=["AAPL"])

#getting the database
db = client.MSI
#Creating a collection
stock_prices = db.stock_prices
#Getting the stock values
prices_msi = []
for x in stock_prices.find({'stock_type': 'COMMON_STOCK'}):
    prices_msi.append(x['stock_price']/72.3)
    
prices_msi = pd.DataFrame(prices_msi,columns=["MSI"])

#getting the database
db = client.Starbucks
#Creating a collection
stock_prices = db.stock_prices
#Getting the stock values
prices_sbux = []
for x in stock_prices.find({'stock_type': 'COMMON_STOCK'}):
    prices_sbux.append(x['stock_price']/72.3)
    
prices_sbux = pd.DataFrame(prices_sbux,columns=["SBUX"])

stock_prices = pd.concat([prices_apple,prices_msi,prices_sbux],axis=1)
#Saving the dataframe

dataset = stock_prices.copy()

#Map Reduce for Momentum

momentum = []
name = ["AAPL" , "MSI" , "SBUX"]
for i in range(3):
    #Getting the dataset
    current = dataset.iloc[-100:,i].values
    current = current.reshape((-1,10))
    
    current = list(
        map(lambda x: 
            reduce(lambda a,b : a*0.9 + (1-0.9)*b , x),#Making the reduce function
            current))                                  #Maping the reduction function
    #Appending the momentum array
    momentum.append(pd.DataFrame(current,columns=[name[i]]))
    #Plotting the momentum
    #plt.plot(current)
    #plt.xlabel("10-Day period")
    #plt.ylabel("Momentum")
    #plt.title(name[i])
    #plt.show()


momentum_data = pd.concat(momentum,axis=1)
momentum_data.to_csv("momentum.csv",index=False)
stock_prices.to_csv("aapl_msi_sbux.csv",index=False)



from keras.models import model_from_json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt

with open('MSI/msi_model.json','r') as f:
    model = model_from_json(f.read())
    
model.load_weights('MSI/msi_w.h5')
scaler = StandardScaler()
dataset = pd.read_csv('aapl_msi_sbux.csv')

#getting a random sample
start = np.random.randint(1000,1030)
sample = dataset.iloc[start:start+30,1].values
sample = scaler.fit_transform(sample.reshape(-1,1))
sample = sample.reshape(sample.shape[0],1,sample.shape[1])

#Getting the forecast
prediction = model.predict(sample)
prediction = scaler.inverse_transform(prediction)

dataframe = pd.DataFrame(list(enumerate(prediction.reshape(-1))),columns = ['Day','Price'])

dataframe.to_csv("msi_forecast.csv")
dataframe.to_json("msi_forecast.json")

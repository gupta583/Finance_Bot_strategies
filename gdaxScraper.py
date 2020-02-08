from Util import gdaxUtil
import pandas as pd

product = 'BTC-USD'
start = '1/1/2016'
end = 'today'
'''Use this script to save data scraped from gdax into a csv file. It will
   save it into the current working directory'''
gdaxUtil.createData(product,start,end,save=True,timestep='daily')
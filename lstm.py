import os
import pandas as pd
from Util.Preprocess import PreProcessor
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
from keras.models import Model

'''Creates LSTM and evaluates it on how well it predicts daily closing price
   Assumes there is a file with OHLCV data in current working directory
   If dataset does not exists refer to gdaxScraper.py'''

'''Specify filepath, should be in the directory of lstm.py'''


'''Function to return PreProcessor instance. Needed to for many preprocessing operations
   if no path is supplied then will use the default filepath:BTC-USD:2015-01-01:2017-10-08:daily.csv
   minfactor and maxfactor arguments are used to estimate future max price of data refer to powerpoint
   slides for more information.
   Args: path (.csv) 
         minfactor [0,1]
         maxfactor [1,infinity)
   Return: Preprocessor instance'''

def loaddata(path = None, minfactor = 1, maxfactor = 7):
    if path is None:
        path = os.path.abspath("BTC-USD:2015-01-01:2017-10-08:daily.csv")
    df = pd.read_csv(path,index_col='time',parse_dates=['time'])
    return PreProcessor(df,minfactor,maxfactor)

path = 'ETH-USD-2016-01-01-2017-11-08-daily.csv'
findata = loaddata(path)
findata.generate(drop = True)
findata.dropNA()
processedData = findata.getData(original=False,isReference=True)
processedData.replace([np.inf, -np.inf], np.nan).dropna(how="all",inplace=True)


trainpercent = .8
train , test = findata.split(trainpercent)
trainScaled = findata.normalize(train.copy(),featureRange=(-1,1),type='minmax')
testScaled = findata.normalize(test.copy(),featureRange=(-1,1),type='minmax')

'''Specify lookback window (period)'''
lookback = 2
'''Specify which columns to actually use'''
columns = train.columns
'''Create moving window to train LSTM'''
train_X , train_Y = findata.createWindow(trainScaled[columns],look_backX=lookback)
test_X, test_Y = findata.createWindow(testScaled[columns],look_backX=lookback)



neurons = [100,1]
epochs = 600; batch = 128
lossfunc = 'mae'; optimizer = 'adam'

'''Function build model, modify by adding layers to function and specify neurons
   Takes additional argument previousModel
   
   Argument shape is very import, training set must have shape of dimensions (samples,lookback,numfeatures)
   Example if we have timeseries of 1-d (numfeatures = 1) like [1,2,3,4,5], with lookback of 2 and then
   our moving window dataset would look like: for train_X [[1,2],[2,3],[3,4]]
                                                  train_Y       [[3],[4],[5]]
                                                       
   If you wish to build a neural net and add LSTM layer supply instance
   of type Model of pretrained net.'''

def buildModel(shape,neuron,previousModel=None):
    if previousModel is None:
        previous = Input(shape=(shape[1], shape[2]))
    else:
        previous = previousModel.output

    lstmlayer = LSTM(neuron[0],return_sequences=False,recurrent_dropout=.3)(previous)
    outlayer = Dense(neuron[-1],activation='linear')(lstmlayer)

    return Model(previous,outlayer,name='cryptobot')

crypto = buildModel(np.shape(train_X),neurons)
crypto.compile(loss =lossfunc,optimizer=optimizer)
history = crypto.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=epochs, batch_size=batch)

'''Prints plots of:
    -curve fit on training set
    -curve fit on testing set
    -training/testing error'''

def printplots(crypto,findata,training,testing,lookback,title):
    plt.close('all')
    train = training[0]
    test = testing[0]
    train_X = training[1]
    train_Y = training[2]
    test_X = testing[1]
    test_Y = testing[2]

    predict = crypto.predict(train_X)
    predictDF = pd.DataFrame(predict,columns=['close'])
    actualDF = pd.DataFrame(train_Y,columns=['close'])
    trainpredictOriginal = findata.normalizeInverse(predictDF,initial=None,type='minmax')
    trainactualOriginal = findata.normalizeInverse(actualDF,initial=None,type='minmax')
    trainpredictOriginal.index = train.index[lookback:]
    trainactualOriginal.index = train.index[lookback:]

    fig = plt.figure()
    plt.title(title[0])
    plt.plot(trainpredictOriginal,label= 'predicted')
    plt.plot(trainactualOriginal,label = 'actual')
    plt.legend()
    fig.autofmt_xdate()

    predict = crypto.predict(test_X)

    predictDF = pd.DataFrame(predict,columns=['close'])
    actualDF = pd.DataFrame(test_Y,columns=['close'])

    testpredictOriginal = findata.normalizeInverse(predictDF,initial=None,type='minmax')
    testactualOriginal = findata.normalizeInverse(actualDF,initial=None,type='minmax')
    testpredictOriginal.index = test.index[lookback:]
    testactualOriginal.index = test.index[lookback:]
    print(mean_absolute_error(testactualOriginal.values,testpredictOriginal.values))
    fig = plt.figure()
    plt.title(title[1])

    plt.plot(testpredictOriginal, label = 'predicted')
    plt.plot(testactualOriginal, label = 'actual')
    plt.legend(loc='upper left')
    fig.autofmt_xdate()

    plt.figure()
    plt.title('Training and Validation Error')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend(loc='upper left')
    plt.show(block = False)

product = 'BTC'
titles = [product + ' Predicted Prices (Training)',product + ' Predicted Prices (Validation)']
printplots(crypto, findata, [train, train_X, train_Y], [test, test_X, test_Y], lookback,titles)

'''Returns actual and predicted converted back to real daily closing prices
   before they were scaled to range (-1,1)'''
def getactualPrices(crypto,findata,training,testing,lookback):
    train = training[0]
    test = testing[0]
    train_X = training[1]
    train_Y = training[2]
    test_X = testing[1]
    test_Y = testing[2]
    predict = crypto.predict(test_X)

    predictDF = pd.DataFrame(predict, columns=['close'])
    actualDF = pd.DataFrame(test_Y, columns=['close'])

    testpredictOriginal = findata.normalizeInverse(predictDF, initial=None, type='minmax')
    testactualOriginal = findata.normalizeInverse(actualDF, initial=None, type='minmax')
    testpredictOriginal.index = test.index[lookback:]

    testactualOriginal.index = test.index[lookback:]
    return [testpredictOriginal,testactualOriginal]


'''Evaluates metrics on accuracy of forcasting model
   Refer to slides for more details'''
transactionfee = .0025
def performanceMetrics(actual,predicted,test,transactionfee=.0001):
    actualprices = test[lookback - 1:]
    B = transactionfee
    S = B

    returns = 0
    for index in range(len(actualprices) - 1):
        predictedPrice = predicted.iloc[index].close
        actualPrice = actualprices.iloc[index].close
        actualPrice_nextday = actualprices.iloc[index + 1].close
        if predictedPrice > actualPrice:
            returns += (actualPrice_nextday - actualPrice - (B * actualPrice + S * actualPrice_nextday)) / actualPrice

        elif predictedPrice < actualPrice:
            returns += (actualPrice - actualPrice_nextday - (B * actualPrice_nextday + S * actualPrice)) / actualPrice
    actual = actual.values.reshape((1,-1))[0]
    predicted = predicted.values.reshape((1,-1))[0]
    pearson_corr , p_value = pearsonr(actual,predicted)
    theil_U= np.sqrt(mean_squared_error(actual,predicted)) / (np.sqrt(np.sum(np.square(predicted)))  + np.sqrt(np.sum(np.square(actual))) )
    mape = np.mean(np.abs((actual- predicted) / actual))

    metrics = ['R', 'Theil U', 'MAPE','Percent Return']
    return pd.Series([pearson_corr,theil_U,mape,returns],index=metrics)

predicted, actual = getactualPrices(crypto,findata,[train,train_X,train_Y],[test,test_X,test_Y],lookback)
performanceMetrics(actual,predicted,test,transactionfee)


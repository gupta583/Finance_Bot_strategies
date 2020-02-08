import pandas as pd
import numpy as np
"""This library is a Financial Toolkit to be expanded on over time as new features
   will inevitably be added in the future

   For now it will mainly be used for the 12 technical indicators we will be using in
   conjunction with GDAX data.
   """

def addSimpleMAVG(dataframe,columname,period=5,newRow=None,inplace=False):
    if inplace and newRow is None:
        dataframe['MAVG' + period.__str__()] = dataframe[columname].rolling(window = period).mean()
    elif not inplace and newRow is not None:
        col = dataframe[columname]
        col = col[-(period - 1):]
        sma = (col.sum() + newRow['columnname']) / period
        newRow['MVAG' + period.__str__()] = sma
    elif not inplace and newRow is None:
        return dataframe[columname].rolling(window=period).mean()
    else:
        error = ValueError("Cannot add row inplace ! Will be updated in future version")
        raise error
def getNextTimeStamp(timeseries):
    delta = timeseries.index[-1] - timeseries.index[-2]
    return timeseries.index[-1] + delta

def addExpMAVG(dataframe,columname,period=5,newRow=None,inplace=False):
    if inplace and newRow is None:
        dataframe['EMAVG' + period.__str__()] = pd.Series.ewm(dataframe[columname],span = period,adjust=False).mean()
    elif not inplace and newRow is None:
        return pd.Series.ewm(dataframe[columname],span = period,adjust=False).mean()
    elif not inplace and newRow is not None:
        if 'EAVG' + period.__str__() in dataframe.columns:
            alpha = 2/(1+period)
            newRow['EAVG' + period.__str__()] = alpha*newRow[columname] +  dataframe['EMAVG'][-1] *(1-alpha)
        else:
            error = ValueError("MAVG for dataframe does not exist")
            raise error
    else:
        error = ValueError("Cannot add row inplace! Will be updated in future version")
        raise error

def addMACD(dataframe,columname,newRow=None,inplace=False):
    if inplace and newRow is None:
        if 'EMAVG12' in dataframe.columns and 'EMAVG26' in dataframe.columns:
            dataframe['MACD'] = dataframe['EMAVG12'] - dataframe['EMAVG26']
        else:
            dataframe['MACD'] = addExpMAVG(dataframe,columname,period=12) - addExpMAVG(dataframe,columname,period=26)
    elif not inplace and newRow is None:
        if 'EMAVG12' in dataframe.columns and 'EMAVG26' in dataframe.columns:
            return dataframe['EMAVG12'] - dataframe['EMAVG26']
        else:
            return addExpMAVG(dataframe,columname,period=12) - addExpMAVG(dataframe,columname,period=26)
    elif not inplace and newRow is not None:
        if 'EMAVG12' in newRow.index and 'EMAVG26' in newRow.index:
            newRow['MACD'] = newRow['EMAVG12'] - newRow['EMAVG26']
        else:
            newRow['MACD'] = addExpMAVG(dataframe,columname,period=12,newRow=newRow) - addExpMAVG(dataframe,columname,period=26,newRow=newRow)
    else:
        error = ValueError("Cannot add row inplace! Will be updated in future version")
        raise error

def addMomentum(dataframe, columname, period, inplace=False):
    if inplace:
        dataframe['mtm' + period.__str__()] = dataframe[columname].diff(periods=period)
    elif not inplace:
        return dataframe[columname].diff(periods=period)


def addROC(dataframe,columname,period=1,newRow=None,inplace=False):
    if inplace and newRow is None:
        dataframe['roc'] = dataframe[columname].pct_change(periods = period)
    elif not inplace and newRow is None:
        return dataframe[columname].pct_change(periods = period)
    elif not inplace and newRow is not None:
        if dataframe.size > period:
            return (dataframe.iloc[-period] - newRow[columname])/dataframe.iloc[-period]
        else:
            return np.NaN
    else:
        error = ValueError("Cannot add row inplace! Will be updated in future version")
        raise error
def addTypicalPrice(dataframe,newRow=None,inplace=False):
    if inplace and newRow is None:
        dataframe['typicalPrice'] =  (dataframe.close.values + dataframe.low.values + dataframe.high.values)/3
    elif not inplace and newRow is None:
        return (dataframe.close.values + dataframe.low.values + dataframe.high.values)/3

def addCCI(dataframe,period=1,newRow=None,inplace=False):
    typical = pd.Series(addTypicalPrice(dataframe))
    k = .15
    typical = (typical - typical.rolling(window = period).mean())/(k*typical.rolling(window=period).std())
    typical.index = dataframe.index
    dataframe['CCI'] = typical



def addStochMI(dataframe,rangePeriod = 13, innerPeriod = 2, outPeriod = 25,newRow=None,inplace=False):
    hiRan = dataframe.high.rolling(window=rangePeriod).max()
    lowRan = dataframe.low.rolling(window=rangePeriod).min()
    midpoint = dataframe.close - (hiRan + lowRan)/2
    hiloDiff = hiRan - lowRan
    top = pd.Series.ewm(pd.Series.ewm(midpoint,span = innerPeriod).mean(),span=outPeriod).mean()
    bottom = pd.Series.ewm(pd.Series.ewm(hiloDiff,span = innerPeriod).mean(),span=outPeriod).mean()/2
    if inplace:
        dataframe['stochMI'] = top/bottom
    elif not inplace:
        return top/bottom

def addAvgTrRange(dataframe, period=14, inplace=False):
        hilo = dataframe.high.values - dataframe.low.values
        hilo = hilo[1:]
        hiprevClose = np.abs(dataframe.high[1:].values - dataframe.close[:-1].values)
        lowprevClose = np.abs(dataframe.low[1:].values - dataframe.close[:-1].values)
        colnames = ['hilo','hiprevClose','lowprevClose']

        newDF = pd.DataFrame({colnames[0] : hilo, colnames[1]:hiprevClose, colnames[2]:lowprevClose},index=dataframe.index[1:])
        newDF = newDF.max(axis=1)
        newDF.index = dataframe.index[1:]
        temp = pd.Series([np.NaN],index=[dataframe.index[0]])
        newDF = pd.concat([temp,newDF])
        if inplace:
            dataframe['true range'] = newDF.rolling(window=period).mean()
        elif not inplace:
            return newDF.rolling(window=period).mean()

def addbollingerBand(dataframe, colname = 'close', period = 14, sigma = 2):
    MA = addSimpleMAVG(dataframe,columname=colname,period = period,inplace=False)
    STD = dataframe.close.rolling(window=period).std()
    bollingerLower = MA - STD * sigma
    bollingerUpper = MA + STD * sigma
    dataframe['bollmid'] = MA
    dataframe['bollLower'] = bollingerLower
    dataframe['bollUpper'] = bollingerUpper


def addAllIndicators(dataframe,inplace=False):
    if inplace is False:
        df = dataframe.copy(deep=True)
    else:
        df = dataframe
    close = 'close'
    addSimpleMAVG(df, close, period=5, inplace=inplace)
    addSimpleMAVG(df, close, period=10, inplace=inplace)
    addExpMAVG(df, close, period=20, inplace=inplace)
    addbollingerBand(df)
    addMACD(df, close, inplace=inplace)
    addROC(df,close,period=21, inplace = inplace)
    addAvgTrRange(df, period=14, inplace=inplace)
    addStochMI(df,rangePeriod=13,innerPeriod=2,outPeriod=25,inplace=inplace)
    addCCI(df,period=20,inplace=inplace)
    addMomentum(df, close, period=30*6, inplace=True)
    addMomentum(df, close, period=30*12, inplace=True)
    return df
